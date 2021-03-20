
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.envs.box2d.lunar_lander import LunarLanderContinuous

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

class policy (nn.Module):
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

THRESHOLD = 50
NUMUPDATES = 30
GAMMA = 0.5

env = LunarLanderContinuous()
memory = list()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()

        # pick an action
        action = env.action_space.sample()

        # observe state after running the action
        observation, reward, done, info = env.step(action)

        # save it in memory
        memory.append( (observation, reward, done, info, state, action) )

        # if it is time to update
        if len(memory) > THRESHOLD:
            for u in range(NUMUPDATES):
                # randomly sample states from memory
                samples = np.random.choice(memory, 5)

                # loop through sample
                for m in samples:
                    # obtain qval and policy value from critic and actor models, respectively
                    qval = qnet1()
                    pol = pol()

                    # compute targets
                    # reward + gamme * (1-done) * qval
                    target = m[1] + GAMMA * (1-m[2]) * qval

                    # perform gradient descent on qnet

                    # perform gradient descent on policy

                    # update target nets

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
