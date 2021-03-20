
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.envs.box2d.lunar_lander import LunarLanderContinuous

class qnet (nn.Module):
    def __init__ (self):
        super(qnet, self).__init__()

        # state has 8 values, action has 2, therefore the input is of shape 10
        # output is a q value for each action, therefore output is of size (2,)
        self.fcone = nn.Linear(10, 3)
        self.fctwo = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.FloatTensor(x)

        x = self.fcone(x)
        x = F.relu(x)

        x = self.fctwo(x)
        #x = F.relu(x)

        return x

class policy (nn.Module):
    def __init__ (self):
        super(policy, self).__init__()

        # policy takes in state as input, which is of shape (8,)
        # and outputs an action
        self.fcone = nn.Linear(8, 3)
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
NUMUPDATES = 10
GAMMA = 0.5

env = LunarLanderContinuous()
memory = list()

critic = qnet()
actor = policy()

criticOptimizer = torch.optim.SGD(critic.parameters(), lr=0.01, momentum=0.9)
actorOptimizer = torch.optim.SGD(actor.parameters(), lr=0.01, momentum=0.9)

loss = torch.nn.MSELoss()
policyLoss = torch.nn.L1Loss()


for i_episode in range(20):
    state = env.reset()
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
                idx = np.random.randint(0, len(memory), 10)

                # loop through sample
                for i in idx:
                    m = memory[i]

                    # obtain qval and policy value from critic and actor models, respectively
                    # actor gets state as input
                    pol = actor.forward( m[4] )

                    # critic gets state and action
                    qval = critic.forward( np.concatenate( [m[4], m[5]] ) )

                    # compute targets
                    # reward + gamme * (1-done) * qval
                    # reward + gamma * (1-done) * predict_qval of next state with the best action in the next state
                    nextStateAction = torch.cat( [ torch.from_numpy(m[0]), actor.forward(m[0]) ] )
                    target = m[1] + GAMMA * (1-m[2]) * critic.forward( nextStateAction )

                    # perform gradient descent on qnet
                    criticOptimizer.zero_grad()

                    criticLoss = loss(target, qval)
                    criticLoss.backward()

                    criticOptimizer.step()

                    # perform gradient descent on policy
                    actorOptimizer.zero_grad()

                    actorLoss = -1 * policyLoss( pol, torch.Tensor([0, 0]) )
                    actorLoss.backward()

                    actorOptimizer.step()

                    # update target nets

        state = observation

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
