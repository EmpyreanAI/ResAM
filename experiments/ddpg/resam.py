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
from spinup import ddpg_tf1
from b3data.utils.stock_util import StockUtil
from spinup.utils.run_utils import ExperimentGrid


def env_fn():
    """Create the MarketEnv environment function.

    Uses information from the B3 data library. Values must be changed within this function.

    Returns:
        The gym.make of the MarketEnv-v0 with the values specified whitin function.

    """
    import gym_market


    stockutil = StockUtil(['PETR3', 'VALE3', 'ABEV3'], [6,6, 9])
    prices, preds = stockutil.prices_preds(start_year=2014,
                                        end_year=2014, period=11)


    return gym.make('MarketEnv-v0', n_insiders=3, start_money=10000,
                    assets_prices=prices, insiders_preds=preds)


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

        - ac_kwargs:activation (tf): Activation function of the hidden states for the actor-critic neural network. Defaults to tf.nn.relu.

        - ac_kwargs:output_activation (tf): Activation function of the output state for the actor-critic neural network. Defaults to tf.tanh.
 
        - ac_kwargs:hidden_sizes ((int,int)): Ammount of hidden states for the actor-critic neural network. Defaults to (256,256).

    Returns:
        The created experiment grid.

    """
    eg = ExperimentGrid(name=name)

    eg.add('env_fn', env_fn)
    eg.add('seed', 3853)
    # eg.add('steps_per_epoch', 5000)
    # eg.add('epochxs', 20000)
    # eg.add('replay_size', int(1e8))
    # eg.add('gamma', 0.99)
    # eg.add('polyak', 0.995)
    # eg.add('pi_lr', 0.0001)
    # eg.add('q_lr', 0.0001)
    # eg.add('batch_size', 9)
    # eg.add('start_steps', 40000)    
    # eg.add('update_after', 1000)
    # eg.add('update_every', 30)
    eg.add('act_noise', 10)
    # eg.add('num_test_episodes', 10)
    # eg.add('max_ep_len', 1000)
    # eg.add('save_freq', 3)
    eg.add('ac_kwargs:activation', tf.tanh)
    eg.add('ac_kwargs:output_activation', tf.tanh)
    eg.add('ac_kwargs:hidden_sizes', (32, 32))

    return eg

def run_exp(experiment, cpus=1):
    """Run the created experiment.
    
    Args:
        experiment (:obj:`ExperimentGrid`): The grid of experiments to execute.
        cpus (int, optional): Ammount of cpus for the experiment. Defaults to 1.
    
    """
    experiment.run(ddpg_tf1, num_cpu=cpus,  data_dir='../../data')
    

if __name__ == '__main__':
    exp = create_exp_grid("MarketDDPG")
    run_exp(exp)
    
   

    

