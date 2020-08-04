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
import sys
import os
import gym
import json
import tensorflow as tf
from datetime import datetime
from spinup import ddpg_tf1, ppo_tf1, td3_tf1, sac_tf1
from b3data.utils.stock_util import StockUtil
from spinup.utils.run_utils import ExperimentGrid
from rmm import RMM

env_fn_args = {
    '_configs': {
        's_money': 10000,
        'taxes': 0.0,
        'allotment': 100,
        'price_obs': True,
        'reward': 'full',
        'log': 'done'
    },

    '_stocks': ['PETR3'], # 'VALE3', 'ABEV3'
    '_windows': [6], #  6, 9
    '_start_year': 2014,
    '_end_year': 2014,
    '_period': 6,
    '_trends': False,
    '_cap': 50
}

if len(sys.argv) > 1:
    n_ins = int(sys.argv[7])
    env_fn_args = {
        '_configs': {
            's_money': float(sys.argv[1]),
            'taxes': float(sys.argv[2]),
            'allotment': int(sys.argv[3]),
            'price_obs': True if sys.argv[4] == "True" else False,
            'reward': sys.argv[5],
            'log': sys.argv[6]
        },
        '_stocks': sys.argv[8:8+n_ins], # 'VALE3', 'ABEV3'
        '_windows': [int(i) for i in sys.argv[8+n_ins:8+(2*n_ins)]],
        '_start_year': int(sys.argv[8+(2*n_ins)]),
        '_end_year': int(sys.argv[8+(2*n_ins)+1]),
        '_period': int(sys.argv[8+(2*n_ins)+2]),
        '_trends': sys.argv[8+(2*n_ins)+3],
        '_cap': [int(i) for i in sys.argv[8+(2*n_ins)+4:8+(2*n_ins)+(4+n_ins)]],
    }

def env_fn():
    """Create the MarketEnv environment function.

    Uses information from the B3 data library. Values must be changed within this function.

    Returns:
        The gym.make of the MarketEnv-v0 with the values specified whitin function.

    """
    import gym_market
    global env_fn_args
    stockutil = StockUtil(env_fn_args['_stocks'], env_fn_args['_windows'])
    prices, preds = stockutil.prices_preds(start_year=env_fn_args['_start_year'],
                                           end_year=env_fn_args['_end_year'],
                                           period=env_fn_args['_period'])

    trends = RMM.trends_group(env_fn_args['_stocks'], prices, start_month=1,
                              period=env_fn_args['_period'], 
                              mean=False, 
                              cap=env_fn_args['_cap'])

    print(trends)
    
    return gym.make('MarketEnv-v0', assets_prices=prices, insiders_preds=preds,
                     configs=env_fn_args['_configs'])

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

        - ac_kwargs:hidden_sizes ((int,int)): Ammount of hidden states for the actor-critic neural network. Defaults to (256,256).

    Returns:
        The created experiment grid.d

    """


    eg = ExperimentGrid(name=name)

    eg.add('env_fn', env_fn)
    eg.add('seed', 9    , in_name=True)
    eg.add('steps_per_epoch', 1000, in_name=True) # Fixed
    eg.add('epochs', 200, in_name=True) # Fix on 100
    eg.add('replay_size', 500000, in_name=True)
    eg.add('gamma', 0.99, in_name=True)
    eg.add('polyak', 0.995, in_name=True)
    eg.add('pi_lr',  0.0005, in_name=True) #000001
    eg.add('q_lr', 0.0001, in_name=True)
    eg.add('batch_size', 100, in_name=True)
    eg.add('start_steps', 10000, in_name=True) # MUUUUITO IMPORTANTE
    eg.add('update_after', 900, in_name=True)
    # eg.add('update_every', 500, in_name=True)
    eg.add('act_noise', 1.0, in_name=True)
    eg.add('ac_kwargs:hidden_sizes', (16, 16), in_name=True)
    # eg.add('ac_kwargs:activation', tf.nn.tanh, in_name=True)

    return eg

def run_exp(new_env_args=None, cpus=1):
    """Run the created experiment.

    Args:
        experiment (:obj:`ExperimentGrid`): The grid of experiments to execute.
        cpus (int, optional): Ammount of cpus for the experiment. Defaults to 1.

    """
    global env_fn_args
    if new_env_args is not None:
        env_fn_args = new_env_args

    experiment = create_exp_grid("MarketDDPG")
    name = ""
    for s in env_fn_args['_stocks']:
        name += s + '_'
    dir = f'../../data/{name}{env_fn_args["_start_year"]}_{env_fn_args["_end_year"]}_{env_fn_args["_configs"]["price_obs"]}'

    try:
        os.mkdir(dir)
    except:
        pass

    with open(f'{dir}/config.json', 'w+') as fp:
        json.dump(env_fn_args['_configs'], fp)

    experiment.run(ddpg_tf1, num_cpu=cpus, data_dir=dir)


if __name__ == '__main__':
    run_exp(cpus=4)
