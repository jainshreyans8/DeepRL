import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import ipdb
from random import random, sample
from dataclasses import dataclass
from typing import Any

@dataclass
class state_transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class DQNAgent(nn.Module):

    def __init__(self, obs_shape, num_actions):
        super(DQNAgent, self).__init__()
        assert len(obs_shape) == 1

        self.model = self.create_model(obs_shape, num_actions)
        self.num_actions = num_actions


    def create_model(self,obs_shape, num_actions):

        layers = []
        layers.append(nn.Linear(obs_shape[0], 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, num_actions))

        net = nn.Sequential(*layers)

        opt = optim.Adam(net.parameters(), lr = 0.0001)

        return net

    
    def forward(self,x):

        return self.model(x)
    
class ReplayBuffer():

    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    

    def insert(self, sars):
        self.buffer.append(sars)
    

    def sample(self, num_samples):
        assert num_samples<=len(self.buffer)
        return sample(self.buffer, num_samples)

    
if __name__ == '__main__':
    
    env = gym.make('CartPole-v1')

    # observation = env.reset()
    observation = torch.rand((4,))
    print(observation)

    agent = DQNAgent(observation.shape, env.action_space.n)

    print(agent.model.forward(observation))



        
        

