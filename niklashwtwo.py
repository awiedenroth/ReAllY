import logging, os

print("b")
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
        self.layer2 = tf.keras.layers.Dense(20,activation=tf.keras.activations.relu)
        self.layer3 = tf.keras.layers.Dense(20,activation=tf.keras.activations.relu)
        self.layer4 = tf.keras.layers.Dense(output_units)

    def call(self, x_in):

        output = {}
        #x = self.layer(x_in)
        #v = self.layer2(x)
        x=self.layer4(self.layer3(self.layer2(x_in)))
        output["q_values"] = x
        return output


class ModelContunous(tf.keras.Model):
    def __init__(self, output_units=2):

        super(ModelContunous, self).__init__()

        self.layer_mu = tf.keras.layers.Dense(output_units)
        self.layer_sigma = tf.keras.layers.Dense(output_units, activation=None)
        self.layer_v = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        mus = self.layer_mu(x_in)
        sigmas = tf.exp(self.layer_sigma(x_in))
        v = self.layer_v(x_in)
        output["mu"] = mus
        output["sigma"] = sigmas

        return output


if __name__ == "__main__":
    print("a")
    kwargs = {
        "model": MyModel,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 1,
    }
    print("a")

    ray.init(log_to_driver=False)

    print("a")
    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    print("a")
    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 1
    saving_after = 5

    learning_rate=0.001
    gamma=0.85

    print("a")
    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    print("a")
    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True)

    print("a")
    # get initial agent
    agent = manager.get_agent()

    optimizer=tf.keras.optimizers.Adam(learning_rate)
    mse=tf.keras.losses.MeanSquaredError()

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

        itertuple = (data_dict["state"], data_dict["action"], data_dict["reward"], data_dict["state_new"], data_dict["not_done"])
        for state, action, reward, state_new, not_done in zip(*itertuple):

        # TODO: optimize agent
            
            with tf.GradientTape() as tape:
                #weights=agent.model.trainable_variables
                #tape.watch(weights)
                qtarget=tf.cast(reward,tf.float32)+gamma*tf.cast(agent.max_q(state_new),tf.float32)
                index=(state.numpy()[0,0],state.numpy()[0,1])
                output=agent.q_val(state,action)
                #print(qtarget,output)
                loss=mse(qtarget,output)
                gradients=tape.gradient(loss,agent.model.trainable_variables)
                #print(gradients,loss)
            optimizer.apply_gradients(zip(gradients,agent.model.trainable_variables))
            #manager.set_agent(agent.model.trainable_variables)
            if not_done == False:
                break
        # update aggregator
        loss=tf.reshape(loss,(1,))
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)
        # print progress
        print(loss)
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in loss])}   avg env steps ::: {np.mean(time_steps)}"
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
