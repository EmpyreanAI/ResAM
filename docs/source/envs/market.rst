==========
Market Env
==========

.. contents:: Table of Contents

General Info
============

.. automodule:: envs.market_env

Env Info
========

The main characteristics of reinforcement learning are the agent and the environment.
The environment is the medium in which the agent is inserted, it can be modified by the agent or undergo changes by itself.
The agent interacts with the environment in each step of the simulation,
performing an action based on his observation made in the state of the environment.
The agent also receives an of the environment a numerical realization of the impact of his actions at each step,
called a reward or reinforcement. The agent's goal is to maximize his accumulated reward during a episode, called a return.
Reinforcement learning methods are a means for an agent to learn a perform to achieve a goal.

.. autoclass:: envs.market_env.MarketEnv

Observation
===========

A state is the complete description of the environment, there is no information about the environment that is not
reflected by the state. When the information does not fully reflect the environment,
describing it partially, it is called observation. When the agent is able to observe the complete state of the environment,
the environment is called fully observable. On the other hand, when only a part is perceived,
we call the environment partially observable.

.. autofunction:: envs.market_env.MarketEnv._observation_space
.. autofunction:: envs.market_env.MarketEnv._make_observation

Actions
=======

Different environments allow for different types of actions. The set of all actions valid in a given environment
is called the action space. There are environments that have discrete action space,
in which a finite number of movements is available to the agent. Other environments,
such as robotic environments, have space for continuous actions, in which actions are vectors of real values.
Actions are performed by an agent according to their policy.

.. autofunction:: envs.market_env.MarketEnv._action_space
.. autofunction:: envs.market_env.MarketEnv._take_action

Reward
======

The agent's objective is to maximize the accumulated value of the reward over a trajectory. One of the possible forms of return, is a sum of all the rewards obtained, deducted from how long in the future each one was obtained. Such an approach is dominated by discounted infinite horizon returns, and requires an additional term called a discount factor.

.. autofunction:: envs.market_env.MarketEnv._get_reward

Step
====

Main gym environment function. Handles all aspects of the reinforcement learning for a single step.

.. autofunction:: envs.market_env.MarketEnv.step


Auxiliary Functions
===================

Other functions created to help in the development of the environment.

.. autofunction:: envs.market_env.MarketEnv._check_done
.. autofunction:: envs.market_env.MarketEnv._current_price
.. autofunction:: envs.market_env.MarketEnv._full_value
.. autofunction:: envs.market_env.MarketEnv._portfolio_value
.. autofunction:: envs.market_env.MarketEnv._daily_returns
.. autofunction:: envs.market_env.MarketEnv._log_step
