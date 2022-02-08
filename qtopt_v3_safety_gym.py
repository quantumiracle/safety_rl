import safety_gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import gym
import torch.optim as optim
import random
import numpy as np
import json
from IPython.display import clear_output
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class CEM():
    ''' 
    cross-entropy method, as optimization of the action policy 
    '''
    def __init__(self, theta_dim, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.theta_dim = theta_dim
        self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)

    def initialize(self, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.mean = ini_mean_scale*np.ones(self.theta_dim)
        self.std = ini_std_scale*np.ones(self.theta_dim)
        
    def sample(self):
        # theta = self.mean + np.random.randn(self.theta_dim) * self.std
        theta = self.mean + np.random.normal(size=self.theta_dim) * self.std
        return theta

    def sample_multi(self, n):
        theta_list=[]
        for i in range(n):
            theta_list.append(self.sample())
        return np.array(theta_list)


    def update(self, selected_samples):
        self.mean = np.mean(selected_samples, axis = 0)
        # print('mean: ', self.mean)
        self.std = np.std(selected_samples, axis = 0)  # plus the entropy offset, or else easily get 0 std
        # print('std: ', self.std)

        return self.mean, self.std


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QT_Opt():
    def __init__(self, state_dim, action_dim, hidden_dim, replay_buffer, q_lr=3e-4, cem_update_itr=4, select_num=6, num_samples=64):
        self.num_samples = num_samples
        self.select_num = select_num
        self.cem_update_itr = cem_update_itr
        self.replay_buffer = replay_buffer
        self.qnet = QNetwork(state_dim+action_dim, hidden_dim).to(device) # gpu
        self.target_qnet1 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.target_qnet2 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.cem = CEM(theta_dim = action_dim)  # cross-entropy method for updating

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.step_cnt = 0

    def update(self, batch_size, gamma=0.9, soft_tau=1e-2, update_delay=100):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        self.step_cnt+=1

        
        state_      = torch.FloatTensor(state).to(device)
        next_state_ = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predict_q = self.qnet(state_, action) # predicted Q(s,a) value

        # get argmax_a' from the CEM for the target Q(s', a')
        new_next_action = []
        for i in range(batch_size):      # batch of states, use them one by one, to prevent the lack of memory
            new_next_action.append(self.cem_optimal_action(next_state[i]))
        new_next_action=torch.FloatTensor(new_next_action).to(device)

        target_q_min = torch.min(self.target_qnet1(next_state_, new_next_action), self.target_qnet2(next_state_, new_next_action))
        target_q = reward + (1-done)*gamma*target_q_min

        q_loss = ((predict_q - target_q.detach())**2).mean()  # MSE loss, note that original paper uses cross-entropy loss
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update the target nets, according to original paper:
        # one with Polyak averaging, another with lagged/delayed update
        self.target_qnet1=self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
        self.target_qnet2=self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)
    


    def cem_optimal_action(self, state):
        ''' evaluate action wrt Q(s,a) to select the optimal using CEM '''
        cuda_states = torch.FloatTensor(np.vstack([state]*self.num_samples)).to(device)
        self.cem.initialize() # every time use a new cem, cem is only for deriving the argmax_a'
        for itr in range(self.cem_update_itr):
            actions = self.cem.sample_multi(self.num_samples)
            q_values = self.target_qnet1(cuda_states, torch.FloatTensor(actions).to(device)).detach().cpu().numpy().reshape(-1) # 2 dim to 1 dim
            max_idx=q_values.argsort()[-1]  # select one maximal q
            idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
            selected_actions = actions[idx]
            _,_=self.cem.update(selected_actions)
        optimal_action = actions[max_idx]
        return optimal_action
 

    def target_soft_update(self, net, target_net, soft_tau):
        ''' Soft update the target net '''
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def target_delayed_update(self, net, target_net, update_delay):
        ''' delayed update the target net '''
        if self.step_cnt%update_delay == 0:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    param.data 
                )

        return target_net

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path)
        torch.save(self.target_qnet1.state_dict(), path)
        torch.save(self.target_qnet2.state_dict(), path)

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet1.load_state_dict(torch.load(path))
        self.target_qnet2.load_state_dict(torch.load(path))
        self.qnet.eval()
        self.target_qnet1.eval()
        self.target_qnet2.eval()

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.plot(rewards)
    plt.savefig('qt_opt_v2.png')
    # plt.show()
        
def main():
    ############## Hyperparameters ##############
    env_name = "Safexp-PointGoal1-v0"
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000000        # max training episodes
    max_timesteps = 1000        # max timesteps in one episode
    batch_size = 128
    update_timestep = 500      # update policy every n timesteps
    hidden_dim = 512
    lr = 0.0003                 # parameters for Adam optimizer
    random_seed = None
    data_path = f'./data/qtopt_{env_name}'
    log_path = f'./data/{env_name}_qtopt.json'
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    replay_buffer_size = 5e5
    replay_buffer = ReplayBuffer(replay_buffer_size)
    model = QT_Opt(state_dim, action_dim, hidden_dim, replay_buffer, lr)
    
    # logging variables
    avg_length = 0
    time_step = 0
    running_reward = 0
    running_cost = 0
    log = {'time_step':[], 'avg_length': [], 'running_reward': [], 'running_cost': []}
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = model.cem_optimal_action(state)
            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # update if its time
            if time_step % update_timestep == 0 and len(replay_buffer) > batch_size:
                model.update(batch_size)
                model.save_model(data_path)
            running_reward += reward
            running_cost += info['cost']
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = running_reward/log_interval
            running_cost = running_cost/log_interval
            log['time_step'].append(time_step)
            log['avg_length'].append(avg_length)
            log['running_reward'].append(running_reward)
            log['running_cost'].append(running_cost)
            json.dump(log, open(log_path, 'w'))

            print('Episode {} \t Avg length: {} \t Avg reward: {:.3f} \t Avg cost: {}'.format(i_episode, avg_length, running_reward, running_cost))
            running_reward = 0
            running_cost = 0
            avg_length = 0
            
        if i_episode% 10==0:
            plot(running_reward)

if __name__ == '__main__':
    main()
    