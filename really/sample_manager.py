import os, logging
from datetime import datetime
import glob
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ray
import tensorflow as tf
import gridworlds
import gym
import numpy as np
from really.agent import Agent
from really.runner_box import RunnerBox
from really.buffer import Replay_buffer
from really.agg import Smoothing_aggregator
from really.utils import all_subdirs_of


class SampleManager():

    """
    @args:
        model: model Object
        environment: string specifying gym environment or object of custom gym-like (implementing the same methods) environment
        num_parallel: int, number of how many agents to run in parall
        total_steps: int, how many steps to collect for the experience replay
        returns: list of strings specifying what is to be returned by the box
            supported are: 'value_estimate', 'log_prob', 'monte_carlo'
        actin_sampling_type: string, type of sampling actions, supported are 'epsilon_greedy', 'thompson', or 'continous_normal_diagonal'

    @kwargs:
        model_kwargs: dict, optional model initialization specifications
        weights: optional, weights which can be loaded into the agent for remote data collecting
        input_shape: shape or boolean (if shape not needed for first call of model), defaults shape of the environments reset state

        env_config: dict, opitonal configurations for environment creation if a custom environment is used

        num_episodes: specifies the total number of episodes to run on the environment for each runner, defaults to 1
        num_steps: specifies the total number of steps to run on the environment for each runner

        gamma: float, discount factor for monte carlo return, defaults to 0.99
        temperature: float, temperature for thomson sampling, defaults to 1
        epsilon: epsilon for epsilon greedy sampling, defaults to 0.95

        remote_min_returns: int, minimum number of remote runner results to wait for, defaults to 10% of num_parallel
        remote_time_out: float, maximum amount of time (in seconds) to wait on the remote runner results, defaults to None
    """

    def __init__(self, model, environment, num_parallel, total_steps, returns=[], **kwargs):


        self.model = model
        self.environment = environment
        self.num_parallel = num_parallel
        self.total_steps = total_steps
        self.returns = returns
        self.kwargs = kwargs
        self.buffer = None

        # create gym / custom gym like environment
        if isinstance(self.environment, str):
            self.env_instance = gym.make(self.environment)
        else:
            env_kwargs = {}
            if 'env_kwargs' in kwargs.keys():
                env_kwargs = kwargs['env_kwargs']
                self.kwargs.pop('env_kwargs')
            self.env_instance = self.env_creator(self.environment, **env_kwargs)

        # initilize empty datasets aggregator
        self.data = {}
        self.data['action'] = []
        self.data['state'] = []
        self.data['reward'] = []
        self.data['state_new'] = []
        self.data['not_done'] = []

        ## some checkups

        assert self.num_parallel > 0, 'num_parallel hast to be greater than 0!'

        # check action sampling type
        if 'action_sampling_type' in kwargs.keys():
            type = kwargs['action_sampling_type']
            if type not in ['thompson', 'epsilon_greedy', 'continous_normal_diagonal']:
                print(f'unsupported sampling type: {type}. assuming thompson sampling instead.')
                self.kwargs['action_sampling_type'] = 'thompson'

        if not('temperature' in self.kwargs.keys()):
            self.kwargs['temperature'] = 1
        if not('epsilon' in self.kwargs.keys()):
            self.kwargs['epsilon'] = 0.95
        # chck return specifications
        for r in returns:
            if r not in ['log_prob', 'monte_carlo', 'value_estimate']:
                print(f'unsuppoerted return key: {r}')
                if r == 'value_estimate':
                    self.kwargs['value_estimate'] = True
            else: self.data[r] = []

        # check for runner sampling method:
        # error if both are specified
        self.run_episodes = True
        self.runner_steps = 1
        if 'num_episodes' in kwargs.keys():
            self.runner_steps = kwargs['num_episodes']
            if 'num_steps' in kwargs.keys():
                print('Both episode mode and step mode for runner sampling are specified. Please only specify one.')
                raise ValueError
            self.kwargs.pop('num_episodes')
        elif 'num_steps' in kwargs.keys():
            self.runner_steps = kwargs['num_steps']
            self.run_episodes = False
            self.kwargs.pop('num_steps')

        # check for remote process specifications
        if 'remote_min_returns' in kwargs.keys():
            self.remote_min_returns = kwargs['remote_min_returns']
            self.kwargs.pop('remote_min_returns')
        else:
            # defaults to 10% of remote runners, but minimum 1
            self.remote_min_returns = max([int(0.1 * self.num_parallel),1])

        if 'remote_time_out' in kwargs.keys():
            self.remote_time_out = kwargs['remote_time_out']
            self.kwargs.pop('remote_time_out')
        else:
            # defaults to None, i.e. wait for remote_min_returns to be returned irrespective of time
            self.remote_time_out = None

        # # TODO: print info on setup values




    def get_data(self, do_print=False, total_steps=None):

        if total_steps is not None:
            old_steps = self.total_steps
            self.total_steps = total_steps

        not_done = True
        # create list of runnor boxes
        runner_boxes = [RunnerBox.remote(Agent, self.model, self.env_instance, runner_position=i, returns=self.returns, **self.kwargs) for i in range(self.num_parallel)]
        t = 0
        # run as long as not yet reached number of total steps
        while not_done:

            if self.run_episodes:
                ready, remaining = ray.wait([b.run_n_episodes.remote(self.runner_steps) for b in runner_boxes], num_returns=self.remote_min_returns, timeout=self.remote_time_out)
            else:
                ready, remaining = ray.wait([b.run_n_steps.remote(self.runner_steps) for b in runner_boxes], num_returns=self.remote_min_returns, timeout=self.remote_time_out)

            # boxes returns list of tuples (data_agg, index)
            returns = ray.get(ready)
            results = []
            indexes = []
            for r in returns:
                result, index = r
                results.append(result)
                indexes.append(index)

            # store data from dones
            if do_print: print(f'iteration: {t}, storing results of {len(results)} runners')
            not_done = self._store(results)
            # get boxes that are alreadey done
            accesed_mapping = map(runner_boxes.__getitem__, indexes)
            dones = list(accesed_mapping)
            # concatenate dones and not dones
            runner_boxes = dones + runner_boxes
            t += 1

        if total_steps is not None:
            self.total_steps = old_steps

        return self.data

    # stores results and asserts if we are done
    def _store(self, results):
        not_done = True
        # results is a list of dctinaries
        assert self.data.keys() == results[0].keys(), "data keys and return keys do not matach"

        for r in results:
            for k in self.data.keys():
                self.data[k].extend(r[k])

        # stop if enought data is aggregated
        if len(self.data['state']) > self.total_steps:

            not_done = False

        return not_done


    def sample(self, sample_size, from_buffer=True):
        # sample from buffer
        if from_buffer:
            dict = self.buffer.sample(sample_size)
        else:
            # save old sepcification
            old_total_steps = self.total_steps
            # set to sampling size
            self.total_steps = sample_size
            self.get_data()
            dict = self.data
            # restore old specification
            self.total_steps = old_total_steps
        return dict


    def get_agent(self, test=False):

        if test:
            old_e = self.kwargs['epsilon']
            old_t = self.kwargs['temperature']
            self.kwargs['epsilon'] = 0
            self.kwargs['temperature'] = 0.0001

        # get agent specifications from runner box
        runner_box = RunnerBox.remote(Agent, self.model, self.env_instance, runner_position=0, returns=self.returns, **self.kwargs)
        agent_kwargs = ray.get(runner_box.get_agent_kwargs.remote())
        agent = Agent(self.model, **agent_kwargs)

        if test:
            self.kwargs['epsilon'] = old_e
            self.kwargs['temperature'] = old_t

        return agent

    def set_agent(self, new_weights):
        self.kwargs['weights'] = new_weights

    def set_temperature(self, temperature):
        self.kwargs['temperature'] = temperature

    def set_epsilon(self, epsilon):
        self.kwargs['epsilon'] = epsilon

    def initilize_buffer(self, size, optim_keys=['state', 'action', 'reward', 'state_new', 'not_done']):
        self.buffer = Replay_buffer(size, optim_keys)

    def store_in_buffer(self, data_dict):
        self.buffer.put(data_dict)

    def test(self, max_steps, test_episodes=100, evaluation_measure='time', render=False, do_print=False):

        env = self.env_instance
        agent = self.get_agent(test=True)
        #agent.epsilon = 1

        # get evaluation specs
        return_time = False
        return_reward = False

        if evaluation_measure == 'time':
            return_time = True
            time_steps = []
        elif evaluation_measure == 'reward':
            return_reward = True
            rewards = []
        elif evaluation_measure == 'time_and_reward':
            return_time = True
            return_reward = True
            time_steps = []
            rewards = []
        else:
            print(f"unrceognized evaluation measure: {evaluation_measure} \n Change to 'time', 'reward' or 'time_and_reward'.")
            raise ValueError

        for e in range(test_episodes):
            state_new = np.expand_dims(env.reset(), axis=0)
            if return_reward:
                reward_per_episode = []

            for t in range(max_steps):
                if render: env.render()
                state = state_new
                action = agent.act(state)
                # check if action is tf
                if tf.is_tensor(action):
                    action=action.numpy()
                state_new, reward, done, info = env.step(int(action))
                state_new = np.expand_dims(state_new, axis=0)
                if return_reward:
                    reward_per_episode.append(reward)
                if done:
                    if return_time:
                        time_steps.append(t)
                    if return_reward:
                        rewards.append(np.mean(reward_per_episode))
                    break
                if (t == max_steps-1):
                    if return_time:
                        time_steps.append(t)
                    if return_reward:
                        rewards.append(np.mean(reward_per_episode))
                    break

        env.close()

        if return_time & return_reward:
            if do_print:
                print(f"Episodes finished after a mean of {np.mean(time_steps)} timesteps")
                print(f"Episodes finished after a mean of {np.mean(rewards)} accumulated reward")
            return time_steps, rewards
        elif return_time:
            if do_print: print(f"Episodes finished after a mean of {np.mean(time_steps)} timesteps")
            return time_steps
        elif return_reward:
            if do_print: print(f"Episodes finished after a mean of {np.mean(rewards)} accumulated reward")
            return rewards


    def initialize_aggregator(self, path, saving_after=10, aggregator_keys=['loss']):
        self.agg = Smoothing_aggregator(path, saving_after, aggregator_keys)

    def update_aggregator(self, **kwargs):
        self.agg.update(**kwargs)

    def env_creator(self, object, **kwargs):
        return object(**kwargs)

    def save_model(self, path, epoch, model_name='model'):
        time_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        full_path = f"{path}/{model_name}_{epoch}_{time_stamp}"
        agent = self.get_agent()
        print('saving model...')
        agent.model.save(full_path)

    def load_model(self, path, model_name=None):
        if model_name is not None:
            # # TODO:
            print('specific model loading not yet implemented')
        else:
            pass
        # alweys leads the latest model
        subdirs = all_subdirs_of(path)
        latest_subdir = max(subdirs, key=os.path.getmtime)
        print('loading model...')
        model = tf.keras.models.load_model(latest_subdir)
        weights = model.get_weights()
        self.set_agent(weights)
        agent = self.get_agent()
        return agent
