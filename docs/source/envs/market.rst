==========
Market Env
==========

.. contents:: Table of Contents

General Info
============

.. automodule:: envs.market_env

Env Info
========

.. autoclass:: envs.market_env.MarketEnv

Observation
===========

.. autofunction:: envs.market_env.MarketEnv._observation_space
.. autofunction:: envs.market_env.MarketEnv._make_observation

Actions
=======

.. autofunction:: envs.market_env.MarketEnv._action_space
.. autofunction:: envs.market_env.MarketEnv._take_action

Reward
======

.. autofunction:: envs.market_env.MarketEnv._get_reward

Step
====

.. autofunction:: envs.market_env.MarketEnv.step


Auxiliar Functions
==================

.. autofunction:: envs.market_env.MarketEnv._check_done
.. autofunction:: envs.market_env.MarketEnv._current_price
.. autofunction:: envs.market_env.MarketEnv._full_value
.. autofunction:: envs.market_env.MarketEnv._portfolio_value
.. autofunction:: envs.market_env.MarketEnv._daily_returns
.. autofunction:: envs.market_env.MarketEnv._log_step
