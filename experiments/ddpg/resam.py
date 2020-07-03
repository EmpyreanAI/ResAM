""" The experimenter script for the EmpyreanAI's ResAM.

In this experiment is used `OpenAI's gym`_, with the :class:`~envs.market_env.MarketEnv` environment,
together with `OpenAI's Spinning Up`_ DDPG Algorithm, to train and test, using multiple
paramenters, a DDPG agent to allocate resources in the stock market.

All the data provided to the market is fetched from `EmpyreanAI's B3Data`_.

:You Should Know:
    Understanding this docummentation might need the understanding the concepts of epochs, steps and episodes.
    An episode is a simulation of a situation from begining to end and consists of multiple steps.
    An step is one time in the simulation. And an epoch is the grouping of steps, for experimenting purposes.

Note:
    This experiment is not a notebook due to the complexity and time consume of the experiments.

Execution::

    $ python resam.py

.. _OpenAI's gym:
  https://gym.openai.com

.. _OpenAI's Spinning Up:
  https://spinningup.openai.com/

.. _EmpyreanAI's B3Data:
  https://github.com/EmpyreanAI/B3Data


"""
import gym
import tensorflow as tf
from spinup import ddpg_tf1, ppo_tf1, td3_tf1
from b3data.utils.stock_util import StockUtil
from spinup.utils.run_utils import ExperimentGrid
from datetime import datetime

def env_fn():
    """Create the MarketEnv environment function.

    Uses information from the B3 data library. Values must be changed within this function.

    Returns:
        The gym.make of the MarketEnv-v0 with the values specified whitin function.

    """
    import gym_market

    stockutil = StockUtil(['PETR3'], [6]) # 'VALE3', 'ABEV3' , 6, 9
    prices, preds = stockutil.prices_preds(start_year=2014, end_year=2014,
                                          period=11)

    configs = {
        's_money': 10000,
        'taxes': 0.0,
        'allotment': 100,
        'price_obs': True,
        'reward': 'full',
        'log': 'done'
    }

    return gym.make('MarketEnv-v0', assets_prices=prices, insiders_preds=preds, configs=configs)



def create_exp_grid(name):
    """Create a pipeline (or grid) with all desired experiments configurations.

    Values that can be changed on the pipeline:

        - env_fn (gym_env): The gym environment for the experiment.

        - seed (int): Seed of the experiment (does not work). Defaults to 0.

        - steps_per_epoch (int): Ammount of steps in an epoch. Defaults to 4000.

        - epochs (int): Ammount of epochs. Defaults to 100.

        - replay_size (int): Size of the replay buffer of experiences. Defaults to 1000000.

        - gamma (float): Discount factor, closer from zero higher power to current rewards,
          closer to one higher power to future rewards. Defaults to 0.99.

        - polyak (float): Interpolation factor in polyak averaging for target networks. Defaults to 0.995.

        - pi_lr (float): Actor (Policy) learning rate. Defaults to 0.001.

        - q_lr (float): Critic (Q-Value) learning rate. Defaults to 0.001.

        - batch_size (int): Ammount of information feed to the model at once. Defaults to 100.

        - start_steps (int): Ammount of steps that will be taken random actions for exploration. Defaults to 10000.

        - update_after (int): How many steps before starting updating the LossQ, LossPi and QVals. Defaults to 1000.

        - update_every (int): Interval in which the updates are applied, without ratio loss. Defaults to 50.

        - act_noise (float): Noise applied to actions to improve exploration. Only applied after start_steps. Defaults to 0.1.

        - num_test_episodes (int): Number of test episodes. Defaults to 10.

        - max_ep_len (int): Maximum size of one episode. Defaults to 1000.

        - save_freq (int): Frequency in which the model is saved to a file. Defaults to 1.

        - ac_kwargs:hidden_sizes ((int,int)): Ammount of hidden states for the actor-critic neural network. Defaults to (256,256).

    Returns:
        The created experiment grid.

    """
    eg = ExperimentGrid(name=name)

    eg.add('env_fn', env_fn)
    eg.add('seed', 7)
    # eg.add('steps_per_epoch', 5000)
    eg.add('epochs', 10)
    # eg.add('replay_size', int(1e8))
    eg.add('gamma', [0.8, 0.5])
    # eg.add('polyak', 0.995)
    eg.add('pi_lr', 0.000000001) #000001
    eg.add('q_lr', 0.00000001)
    # eg.add('batch_size', 16)
    eg.add('start_steps', 100000) # MUUUUITO IMPORTANTE
    # eg.add('update_after', 2000)
    eg.add('update_every', 1000)
    # eg.add('act_noise', 1)
    # eg.add('num_test_episodes', 10)
    # eg.add('max_ep_len', 1000)
    # eg.add('save_freq', 3)
    # eg.add('ac_kwargs:hidden_sizes', (1024, 1024))

    return eg

def run_exp(experiment, cpus=1):
    """Run the created experiment.
    
    Args:
        experiment (:obj:`ExperimentGrid`): The grid of experiments to execute.
        cpus (int, optional): Ammount of cpus for the experiment. Defaults to 1.
    
    """
    now = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    experiment.run(ddpg_tf1, num_cpu=cpus,  data_dir='../../data/' + now)
    

if __name__ == '__main__':
    exp = create_exp_grid("MarketDDPG")
    run_exp(exp)