import gym
import numpy as np
from numpy import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class qnet (nn.Module):
    def __init__ (self):
        super(qnet, self).__init__()

        self.fcone = nn.Linear(4, 2)
        #self.fctwo = nn.Linear(30, 10)
        #self.fcthree = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.FloatTensor(x)

        x = self.fcone(x)
        x = F.relu(x)

        #x = self.fctwo(x)
        #x = F.relu(x)

        #x = self.fcthree(x)
        #x = F.relu

        return x


env = gym.make('CartPole-v0')
obs = env.reset()
#renderedenv = env.render().reshape(1, -1)
qnetwork = qnet()

alpha = 0.1
gamma = 0.6
epsilon = 0.1

epochs = []
penalties = []

for i_episode in range(20):
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax( qnetwork.forward( obs ).numpy() )

        nextobservation, reward, done, info = env.step(action)

        oldq = np.max( qnetwork.forward( obs ).item() )
        nextq = np.max( qnetwork.forward( nextobservation ).item() )

        newval = (1 - alpha) * oldq + alpha * (reward + gamma * nextq)

        qnetwork.zero_grad()
        nextq.backward(newval)


        if reward == -10:
            penalties += 1

        renderedenv = nextrenderedenv
        epochs += 1

print(f"Episode: {epochs}")

env.close()
