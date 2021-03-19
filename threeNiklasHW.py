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



class QModel(tf.keras.Model):
    def __init__(self, output_units=1):

        super(QModel, self).__init__()
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

class PiModel(tf.keras.Model):
    def __init__(self, output_units=2):

        super(PiModel, self).__init__()
        self.layer2 = tf.keras.layers.Dense(32,activation=tf.keras.activations.tanh)
        self.layer3 = tf.keras.layers.Dense(32,activation=tf.keras.activations.tanh)
        self.layer4 = tf.keras.layers.Dense(output_units*2)

    def call(self, x_in):
        output = {}
        #x = self.layer(x_in)
        #v = self.layer2(x)
        x=self.layer4(self.layer3(self.layer2(x_in)))
        output["action"] = x
        return output

class MyModel(tf.keras.Model):
    def __init__(self, output_units=2):

        super(MyModel, self).__init__()
        self.q=QModel()
        self.pi=PiModel()

    def call(self, x_in):
        #print(x_in)
        output={}
        a = self.pi(x_in)
        output["mu"]=tf.slice(a["action"],[0,0],[1,2])
        output["sigma"]=tf.math.exp(tf.slice(a["action"],[0,2],[1,2]))
        #print(a)
        output["q_values"]=self.q(tf.concat([a["action"],x_in],axis=1))["q_values"]
        #print(output)
        return output
        

    def max_q(self, x):
        # computes the maximum q-value along each batch dimension
        model_out = self(x)
        print(model_out)
        x = tf.reduce_max(model_out["mu"], axis=-1)
        return x

if __name__ == "__main__":
    kwargs = {
        "model": MyModel,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "continous_normal_diagonal",
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
    p_=0.99
    
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
    manager.test(test_steps, do_print=True, evaluation_measure="reward")

    # get initial agent
    agent = manager.get_agent()
    import copy
    agentTarget = copy.deepcopy(agent)
    optimizer_q=tf.keras.optimizers.Adam(learning_rate)
    optimizer_pi=tf.keras.optimizers.Adam(learning_rate)
    mse=tf.keras.losses.MeanSquaredError()
    epsilon=1
    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size,from_buffer=True)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")
        #print(agent.model.trainable_variables[0])
        itertuple = (data_dict["state"], data_dict["action"], data_dict["reward"], data_dict["state_new"], data_dict["not_done"])
        lossSum=0
        count=0
        for state, action, reward, state_new, not_done in zip(*itertuple):
            
            with tf.GradientTape(persistent=True) as tapeq:#, tf.GradientTape as tapepi:
                #manager.set_agent(agentTarget)
                qtarget=tf.cast(reward,tf.float32)+gamma*tf.cast(not_done,tf.float32)*tf.cast(agent.max_q(state_new),tf.float32)
                #qtarget=tf.cast(reward,tf.float32)+gamma*tf.cast(not_done,tf.float32)*tf.cast(agentTarget.max_q(state_new),tf.float32)
                #manager.set_agent(agent)

                index=(state.numpy()[0,0],state.numpy()[0,1])
                output,=agent.q_val(state,[[0]])
                loss=mse(qtarget,output)
                gradients_q=tapeq.gradient(loss,agent.model.q.trainable_variables)
                optimizer_q.apply_gradients(zip(gradients_q,agent.model.q.trainable_variables))
                gradients_pi=tapeq.gradient(-output,agent.model.pi.trainable_variables)
                optimizer_pi.apply_gradients(zip(gradients_pi,agent.model.pi.trainable_variables))
                lossSum+=loss
                count+=1
                #print(gradients,loss)

            #manager.set_agent(agent.model.get_weights())
            #agentTarget.model.trainable_variables.assign(agent.model.trainable_variables)
            #paramsTarget=p_ * paramsTarget + (1-p_)*agent.model.trainable_variables
        # update aggregator
        loss=lossSum/count
        loss=tf.reshape(loss,(1,))
        time_steps = manager.test(test_steps, evaluation_measure="reward")
        manager.update_aggregator(loss=loss, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in loss])}   avg reward ::: {np.mean(time_steps)}"
        )

        #manager.set_agent(agent.model.get_weights())

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
