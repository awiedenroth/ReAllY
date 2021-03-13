import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
from really.utils import dict_to_dict_of_datasets
import tensorflow as tf

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space):
        self.action_space = action_space
        ## # TODO:
        self.qtable={}
        for i in range(h):
            for j in range(w):
                self.qtable[(i,j)]=np.zeros((1,self.action_space))


    def __call__(self, state):
        ## # TODO:
        state=np.squeeze(state)
        output = {}
        output["q_values"]=self.qtable[(state[0],state[1])]
        return output

    # # TODO:
    def get_weights(self):
        return self.qtable

    def set_weights(self, q_vals):
        self.qtable=q_vals
        pass

    # what else do you need?
    def max_q(self,x):
        return np.reduce_max(self.__call__(x)["q_values"],axis=1)

    def q_val(self,x,actions):
        return np.take(self.__call__(x)["q_values"],actions,axis=1)

    #def save(self, path):


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs
        # and more
    }

    # initilize
    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)

    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 1#8
    saving_after = 5

    alpha=0.2
    gamma=0.85

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict= dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")

        # TODO: iterate through your datasets
        itertuple = (data_dict["state"], data_dict["action"], data_dict["reward"], data_dict["state_new"], data_dict["not_done"])
        for state, action, reward, state_new, not_done in zip(*itertuple):

        # TODO: optimize agent
            weights=agent.model.get_weights()
            loss=tf.cast(reward,tf.float64)+gamma*agent.max_q(state_new)-agent.q_val(state,action)
            index=(state.numpy()[0,0],state.numpy()[0,1])
            weights[index][0,action.numpy()] += (alpha * loss).numpy()
            manager.set_agent(weights)
            if not_done == False:
                break



        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)
        # print progress
        print(loss)
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in loss])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        manager.set_epsilon(epsilon=0.99)

        #if e % saving_after == 0:
            # you can save models
            #manager.save_model(saving_path, e)

    # and load models
    #manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=30, render=True)
