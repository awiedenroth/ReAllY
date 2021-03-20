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
        self.layer2 = tf.keras.layers.Dense(32,activation=tf.keras.activations.tanh)
        self.layer3 = tf.keras.layers.Dense(32,activation=tf.keras.activations.tanh)
        self.layer4 = tf.keras.layers.Dense(output_units)

    def call(self, x_in):

        output = {}
        #x = self.layer(x_in)
        #v = self.layer2(x)
        x=self.layer4(self.layer3(self.layer2(x_in)))
        output["q_values"] = x
        return output




if __name__ == "__main__":
    #most of the stuff below as in gridworld/showcase
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
    epochs = 100
    sample_size = 1000
    optim_batch_size = 1
    saving_after = 5

    learning_rate=0.001
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

    optimizer=tf.keras.optimizers.Adam(learning_rate)
    mse=tf.keras.losses.MeanSquaredError()
    epsilon=1 #we'll multiply by 0.9 in each epoch
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
        lossSum=0
        count=0
        for state, action, reward, state_new, not_done in zip(*itertuple):
            
            with tf.GradientTape() as tape:
                
                if not_done:
                    qtarget=tf.cast(reward,tf.float32)+gamma*tf.cast(agent.max_q(state_new),tf.float32)
                else:
                    qtarget=tf.cast(reward,tf.float32)
                index=(state.numpy()[0,0],state.numpy()[0,1])
                output=agent.q_val(state,action)
                
                loss=mse(qtarget,output)
                gradients=tape.gradient(loss,agent.model.trainable_variables)
                lossSum+=loss
                count+=1
                #print(gradients,loss)
            optimizer.apply_gradients(zip(gradients,agent.model.trainable_variables))
        manager.set_agent(agent.model.get_weights())
           
        # update aggregator
        loss=lossSum/count
        loss=tf.reshape(loss,(1,))
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in loss])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        epsilon*=0.9
        manager.set_epsilon(epsilon)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
