import gym
import time
import dqn_agent
import torch

env = gym.make('CartPole-v1')
observation = env.reset()

observation = torch.from_numpy(observation)
rand_tensor = torch.rand(observation.shape)
print("observation: ", observation)
print("random tensor: ", rand_tensor)

agent = dqn_agent.DQNAgent(observation.shape, env.action_space.n)

print(agent.model.forward(observation.float()))
