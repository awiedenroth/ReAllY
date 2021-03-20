import gym
import numpy as np
from numpy import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class qnet (nn.Module):
    def __init__ (self):
        super(qnet, self).__init__()

        self.fcone = nn.Linear(4, 3)
        self.fctwo = nn.Linear(3, 2)
        #self.fcthree = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.FloatTensor(x)

        x = self.fcone(x)
        x = F.relu(x)

        x = self.fctwo(x)
        #x = F.relu(x)

        #x = self.fcthree(x)
        #x = F.relu

        return x


env = gym.make('CartPole-v0')

qnetwork = qnet()
optimizer = torch.optim.SGD(qnetwork.parameters(), lr=0.01, momentum=0.9)
loss = torch.nn.MSELoss()

alpha = 0.09
gamma = 0.9
epsilon = 1

epochs = []
penalties = []
counts = []
losses = []

for i_episode in range(100):
    epochs, penalties, reward = 0, 0, 0

    obs = env.reset()
    #env.render()
    done = False
    count = 0

    while not done:
        count += 1

        if random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax( qnetwork.forward( obs ) ).item()

        nextobservation, reward, done, info = env.step(action)

        oldq = qnetwork.forward( obs )
        nextq = qnetwork.forward( nextobservation )

        newval = (1 - alpha) * oldq + alpha * (reward + gamma * nextq)

        l = loss(oldq, newval)
        #optimizer.zero_grad()
        l.backward()

        if reward == -10:
            penalties += 1

        #print(obs)
        obs = nextobservation
        epochs += 1

    optimizer.step()
    epsilon = epsilon * 0.9
    counts.append(count)
    #print(f"Episode: {i_episode} Count: {count}")

'''
obs = env.reset()
env.render()
while True:
    action = torch.argmax( qnetwork.forward( obs ) ).item()
    env.step(action)
'''

import matplotlib.pyplot as plt
plt.plot(losses)
plt.title( f"Average Number of Steps: {sum(counts)/len(counts)}" )
plt.show()


env.close()
