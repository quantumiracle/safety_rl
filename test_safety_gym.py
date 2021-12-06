import safety_gym
import gym

env = gym.make('Safexp-PointGoal1-v0')
print(env.observation_space, env.action_space)

def rollout(env):
    observation = env.reset()
    for _ in range(10000):
        action = 0
        next_observation, reward, done, info = env.step(action)
        print(info)
        env.render()

if __name__ == "__main__":
    rollout(env)