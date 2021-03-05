import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


class MyModel(tf.keras.Model):
    def __init__(self, output_units=2):

        super(MyModel, self).__init__()
        self.layer0 = tf.keras.layers.Flatten(input_shape=(8,1))
        self.layer = tf.keras.layers.Dense(output_units)
        self.layer2 = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        x = self.layer0(x_in)
        x = self.layer(x)
        v = self.layer2(x)
        output["q_values"] = x
        return output


class ModelContunous(tf.keras.Model):
    def __init__(self, output_units=2):

        super(ModelContunous, self).__init__()

        self.layer_mu = tf.keras.layers.Dense(output_units)
        self.layer_sigma = tf.keras.layers.Dense(output_units, activation=None)
        self.layer_v = tf.keras.layers.Dense(1)

    def call(self, x_in):

        sigmas = tf.exp(self.layer_sigma(x_in))
        v = self.layer_v(x_in)
        return v
        '''
        output = {}
        mus = self.layer_mu(x_in)
        output["mu"] = mus
        output["sigma"] = sigmas

        return output
        '''

if __name__ == "__main__":

    kwargs = {
        "model": MyModel,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 1,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

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
    print(agent.model.trainable_variables)

    model = MyModel(8)

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
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")

        # TODO: iterate through your datasets
        model = MyModel()

        for ac, st, r, sn, d in zip(data_dict["action"], data_dict["state"], data_dict["reward"], data_dict["state_new"], data_dict["not_done"]):
            pred = model(st)

            # TODO: optimize agent
            model.optimize(
                    optimizer = tf.keras.optimizers.Adam(0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

            model.fit(
                    dataset,
                    epoch=1
            )

        dummy_losses = [
            np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        ]

        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in dummy_losses])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
