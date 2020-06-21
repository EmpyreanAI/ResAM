""" The experimenter script for the Pendulum Environment.

In this experiment is used `OpenAI's gym`_, with the `Pendulum environment`_,
together with `OpenAI's Spinning Up`_ DDPG Algorithm, to train and test a DDPG agent
to balance a Pendulum.

**Why is this here?**

    The pendulum environment is used to test the performance of the DDPG and compare
    with the ResAM Env.

Note:
    This experiment is not a notebook due to the complexity and time consume of the experiments.


Execution::

    $ python pendulum.py
    $ python pendulum.py --help

.. _OpenAI's gym:
  https://gym.openai.com

.. _OpenAI's Spinning Up:
  https://spinningup.openai.com/

.. _Pendulum environment:
  https://gym.openai.com/envs/Pendulum-v0/


"""

import gym
import argparse
from spinup import ddpg_tf1 as ddpg
from spinup.utils.run_utils import setup_logger_kwargs


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='Pendulum-v0')
        parser.add_argument('--hid', type=int, default=256)
        parser.add_argument('--l', type=int, default=2)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--seed', '-s', type=int, default=0)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--exp_name', type=str, default='ddpg')
        args = parser.parse_args()

        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

        ddpg(lambda : gym.make(args.env),
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                logger_kwargs=logger_kwargs)