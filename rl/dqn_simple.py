
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import random, numpy, argparse, logging, os
from collections import namedtuple
from torch.autograd import Variable
import numpy as np
import datetime

GAMMA = 0.99
EPSILON = 0.6
LR = 1e-4                  # learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, act_shape, obs_shape, out_channels=8, kernel_size=5, stride=1, hidden_units=256):
        super(QNetwork, self).__init__()
        in_dim = obs_shape[0]
        out_dim = act_shape

        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_dim)
        )

        self.linear.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        o = self.linear(x.view(x.size(0), -1))
        return o

class QNetworkCNN(nn.Module):
    def __init__(self, num_actions, in_shape, out_channels=8, kernel_size=5, stride=1, hidden_units=256):
        super(QNetworkCNN, self).__init__()

        self.in_shape = in_shape
        in_channels = in_shape[0]
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels/2), kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size, stride=2),
            nn.Conv2d(int(out_channels/2), int(out_channels), kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size, stride=2)
        )
        self.conv.apply(self.init_weights)

        self.linear = nn.Sequential(
            nn.Linear(self.size_after_conv(), hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )

        self.linear.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def size_after_conv(self,):
        x = torch.rand(1, *self.in_shape)
        o = self.conv(x)
        size=1
        for i in o.shape[1:]:
            size*=i
        return int(size)

    def forward(self, x):
        x = self.conv(x)
        o = self.linear(x.view(x.size(0), -1))
        return o

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, samples):
        # Append when the buffer is not full but overwrite when the buffer is full
        wrap_tensor = lambda x: torch.tensor([x])
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*map(wrap_tensor, samples)))
        else:
            self.buffer[self.location] = transition(*map(wrap_tensor, samples))

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQN(object):
    def __init__(self, action_shape, obs_shape):
        self.eval_net, self.target_net = QNetwork(action_shape, obs_shape).to(device), QNetwork(action_shape, obs_shape).to(device)
        self.action_shape = action_shape
        self.learn_step_counter = 0                                     # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
        else:   # random
            action = np.random.randint(0, self.action_shape)
        return action

    def learn(self, sample,):
        # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
        batch_samples = transition(*zip(*sample))

        # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
        states = torch.cat(batch_samples.state)
        next_states = torch.cat(batch_samples.next_state)
        actions = torch.cat(batch_samples.action)
        rewards = torch.cat(batch_samples.reward)
        is_terminal = torch.cat(batch_samples.is_terminal)
        # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
        # Note: policy_network output Q-values for all the actions of a state, but all we need is the A_t taken at time t
        # in state S_t.  Thus we gather along the columns and get the Q-values corresponds to S_t, A_t.
        # Q_s_a is of size (BATCH_SIZE, 1).
        Q = self.eval_net(states) 
        Q_s_a=Q.gather(1, actions)

        # Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
        # Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
        # values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
        # to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

        # Get the indices of next_states that are not terminal
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        # Select the indices of each row
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

        Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
        if len(none_terminal_next_states) != 0:
            Q_s_prime_a_prime[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the target
        target = rewards + GAMMA * Q_s_prime_a_prime

        # Huber loss
        loss = f.smooth_l1_loss(target, Q_s_a)
        # Zero gradients, backprop, update the weights of policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, model_path=None):
        torch.save(self.eval_net.state_dict(), 'model/dqn')

    
# if __name__ == '__main__':
#     env = bigwaterlemon(headless=False)  
#     dqn = DQN(env.action_shape, env.obs_shape)
#     r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
#     log = []
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
#     print('\nCollecting experience...')
#     for epi in range(MAX_EPI):
#         s=env.reset()
#         epi_r = 0
#         for step in range(MAX_STEP):
#             a = dqn.choose_action(s)
#             s_, r, done = env.step(a)
#             r_buffer.add(torch.tensor([s]), torch.tensor([s_]), torch.tensor([[a]]), torch.tensor([[r]], dtype=torch.float), torch.tensor([[done]]))
#             epi_r += r
#             if step > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
#                 sample = r_buffer.sample(BATCH_SIZE)
#                 dqn.learn(sample)

#             if done:
#                 break
#             s = s_
#         print('Ep: ', epi, '| Ep_r: ', epi_r, '| Steps: ', step)
#         log.append([epi, epi_r, step])
#         if epi % SAVE_INTERVAL == 0:
#             torch.save(dqn.eval_net.state_dict(), 'model/dqn')
#             np.save('log/'+timestamp, log)