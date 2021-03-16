import gym
import numpy as np
from numpy import random
import torch

env = gym.make('CartPole-v0')

qtable = np.zeros( (200, env.action_space.n) )

alpha = 0.1
gamma = 0.6
epsilon = 0.1

epochs = []
penalties = []

### Logistic Regression Start

from sklearn.linear_model import LinearRegression
actionone = LinearRegression()
actionone.fit([[0, 0, 0, 0]], [0])
actionzero = LinearRegression()
actionzero.fit([[0, 0, 0, 0]], [0])

### Logistic Regression End

def discretizestate (obs):
    obs[0] = np.round(obs[0] / 1.92)
    obs[1] = int(obs[1] > 0)
    obs[2] = np.round(obs[2] / 0.1672)
    obs[3] = int(obs[3] > 0)

    return obs.astype(int)

for i_episode in range(20):
    observation = env.reset()
    observation = [observation]

    env.render()
    #observation = discretizestate(observation)
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            #action = int( np.argmax( qtable[observation] ) )
            actionreg = actionzero if actionzero.predict(observation) > actionone.predict(observation) else actionone
            action = 0 if actionzero.predict(observation) > actionone.predict(observation) else 1

        nextobservation, reward, done, info = env.step(action)
        #nextobservation = discretizestate(nextobservation)

        #oldq = qtable[observation, action]
        oldq = actionreg.predict(observation)
        #nextq = np.max( qtable[observation] )
        nextq = np.max([actionzero.predict( observation ), actionone.predict( observation )])

        newval = (1 - alpha) * oldq + alpha * (reward + gamma * nextq)
        #qtable[observation, action] = newval
        actionreg.fit(observation, newval)


        if reward == -10:
            penalties += 1

        observation = [nextobservation]
        epochs += 1

print(f"Episode: {epochs}")

env.close()
