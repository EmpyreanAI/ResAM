"""An Gym environment for Resource Allocation on the market.

This module is indended to be installed as a library and be used ony with the `Gym library`_.
To install it go to the gym_market folder and execute the following command::

    $ pip install -e .

After that, the library can be used in other scripts as follows::

    $ gym.make('MarketEnv-v0', n_insiders=3, start_money=10000,
                assets_prices=prices, insiders_preds=preds)

.. _Gym library:
  https://gym.openai.com

"""

import gym
import random
import numpy as np
from gym import spaces
from collections import deque

class MarketEnv(gym.Env):
    """An Gym environment for Resource Allocation on the market.

    Args:
        n_insiders (int): Number of insiders in the env (similar to number of different assets).
        start_money (float): Ammount of money the agent will have at the start.
        assets_prices (matrix): Price of each asset for each timestep
        insiders_preds (matrix): Prediction of each asset for each timestep

    Attributes:
        start_money (float): Ammount of money the agent will have at the start.
        balance (float): The ammout of maney the agent have at timestep.
        n_insiders (int): Number of insiders in the env.
        assets_prices (matrix): Price of each asset for each timestep.
        insiders_preds (matrix): Prediction of each asset for each timestep.
        epoch_profit (float): Profit of the epoch.
        episode (int): Number of the episode.
        ep_step (int): Number of the step in the episode.
        ep_ret (float): Return of the episode.
        pre_state (vector): Previous state.
        action_space (vector): Action space.
        observation_space: (vector) Observation space.
        full_taxes (float): Ammount of taxes spent in a step.
        taxes (float): Cost of the trading tax.
        min_allotment (int): Minimum ammount of stocks that the agent can buy.
        shares (dict):  A dict of dicts of vectors, representing the amount of shares for each bought price for each asset.

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, n_insiders, start_money, assets_prices, insiders_preds):
        self.start_money = start_money
        self.balance = start_money
        self.n_insiders = n_insiders
        self.assets_prices = assets_prices
        self.insiders_preds = insiders_preds

        self.epoch_profit = 0

        self.episode = 0
        self.ep_step = -1
        self.ep_ret = 0

        self.pre_state = []
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.full_taxes = 0
        self.taxes = 0
        self.min_allotment = 100
        self.shares = {}
        for i in range(0, n_insiders):
            self.shares[i] = {}

    def _observation_space(self):
        """Define the observation space of the environment.

        Returns:
            A spaces.Box() value with the observation space.

        """
        money_low = [0, 0, 0, 0, 0]
        money_high = [np.inf, np.inf, 1, 1, 1]
        money = spaces.Box(np.array(money_low), np.array(money_high))
        return money

    def _action_space(self):
        """Define the action space of the environment.

        The current selected action space is a vector of values between -1 and 1 where:
            - [-1, 0) = Buy x percentage of the available resources in this stock;
            - 0 = Do noting;
            - 1 = Sell Everything.

        Returns:
            A spaces.Box() value with the action space.
        """
        action_low = [-1, -1, -1]
        action_high = [1, 1, 1]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _daily_returns(self):
        """Calculate the appreciation of the portfolio in the current day.

        The appreciation based on the bought price.

        Returns:
            The total appreciation.

        """
        day_ret = 0
        for share_index in self.shares:
            share = self.shares[share_index]
            day_price = self._current_price(share_index)
            for holding in share:
                holding_amt = share[holding]
                day_ret += (day_price - holding)*holding_amt
        return day_ret

    def _portfolio_value(self):
        """Calculate the agents portfolio worth at the time-step.

        Returns:
            The portfolio worth.
        """
        value = 0
        for share_index in self.shares:
            share = self.shares[share_index]
            for holding in share:
                value += share[holding]*self._current_price(share_index)
        return value

    def _full_value(self):
        """Get how much money the agent have in total.

        The money considers the sum of the avaiable money in the balance 
        and the value of the agent's portifolio at the given time-step.

        Returns:
            The sum of money the agent have in the balance and in its portifolio.

        """
        value = self._portfolio_value()
        value += self.balance
        return value

    def _current_price(self, asset):
        """Get the current price for a given asset.

        The price is obtained considering the time-step of the simulation.

        Args:
            asset (int): Index of the asset in the vector to get the price.

        Returns:
            The current price of the asset at a time-step.

        """
        return self.assets_prices[asset][self.ep_step]

    def step(self, action):
        """Run one timestep of the environment’s dynamics.
         
        When end of episode is reached, you are responsible for calling reset() to reset this environment’s state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (object): an action provided by the agent.

        Returns:
            observation (object): agent’s observation of the current environment.

            reward (float) : amount of reward returned after previous action.

            done (bool): whether the episode has ended, in which case further step() calls will return undefined results.
            
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """
        self.ep_step += 1
        action = [1 if x > .5 else 0 if x > 0 else x for x in action]
        self._take_action(action)
        self.state = self._make_observation()
        reward = self._get_reward()
        done = self._check_done()
        info = self._get_info()
        # self._log_step(True, action, reward, done)
        self.pre_state = self.state
        return self.state, reward, done, info

    def _check_done(self):
        """Check if the episode is done.

        The episode is done in two cases. Either the dataset is over, or the full ammount
        of the agent's value is less than 80% of the beginning value.

        Returns:
            A boolean value indicating if it is over or not.
        """
        bankrupcy = self._full_value() < 0.8*self.start_money
        data_end = len(self.assets_prices[0])-1 == self.ep_step
        if bankrupcy or data_end:
            return True
        else:
            return False

    def _normalize(self, actions):
        actions_normal = [0 if x > 0 else abs(x) for x in actions]
        total = sum(actions_normal)
        if total > 1:
            actions_normal = [x/total for x in actions_normal]
        return actions_normal

    def _sell_action(self, actions):
        """Execute all sell actions of a time-step.

        This function handles the selling action.

        Args:
            actions ([float]): Vector of actions between -1 and 1,
            each index for a given asset.
            
        """
        self.sold_profit = 0
        for share_index, action in enumerate(actions):
            num_sold = 0
            if action == 1:
                day_price = self._current_price(share_index)
                share = self.shares[share_index]
                for holding in share:
                    self.sold_profit += (day_price - holding)*share[holding]
                    num_sold += share[holding]
                self.balance += (day_price*num_sold)
                self.balance -= self.taxes
                self.taxes_reward -= self.taxes
                share.clear()

    def _buy_action(self, actions):
        """Execute all buy actions of a time-step.

        This function handles the buy action.

        Args:
            actions ([float]): Vector of actions between -1 and 1,
            each index for a given asset.

        """
        money = self.balance
        for share_index, action in enumerate(actions):
            if action > 0:
                day_price = self._current_price(share_index)
                allotment_price = day_price*self.min_allotment
                alloc_money = round(action*money, 2)
                allot_buy_amt = alloc_money // allotment_price
                total_shares = allot_buy_amt * self.min_allotment
                if total_shares > 0:
                    if day_price in self.shares[share_index]:
                        self.shares[share_index][day_price] += total_shares
                    else:
                        self.shares[share_index][day_price] = total_shares
                self.balance -= (total_shares*day_price)
                self.balance -= self.taxes
                self.taxes_reward -= self.taxes

    def _take_action(self, actions):
        """Execute the actions vectors in the env for the step.

        All the sell actions are made first, so the agent can allocate the money.
        After this is taken care of, the actions left are normalized to execute the buy,
        which allocate the resources of the wallet in the assets.

        Note:
            All agent's profits and expenses are updated in this function. 
            
            On buying:
                - Taxes are paid;
                - Money is decreased;
                - Asset is accounted to the portfolio.
            
            On sell:
                - Taxes are paid;
                - Money is increased;
                - Asset is removed from the portfolio.

        Args:
            actions ([floar]): Vector of actions between -1 and 1,
            each index for a given asset.

        """
        self.taxes_reward = 0
        self._sell_action(actions)
        actions_normal = self._normalize(actions)
        self._buy_action(actions_normal)


    def _make_observation(self):
        """Make observations about the env.

        Observatios contains:
            - Balance;
            - Portifolio Value;
            - Prediction for each asset.

        Returns:
            The observation for the timestep.

        """
        obs = np.zeros(self.observation_space.shape)
        obs[0] = self.balance
        obs[1] = self._portfolio_value()
        obs[2] = self.insiders_preds[0][self.ep_step]
        obs[3] = self.insiders_preds[1][self.ep_step]
        obs[4] = self.insiders_preds[2][self.ep_step]

        return obs

    def _get_reward(self):
        """Calculate the reward for the step.

        The reward value is the sell provit minus all the taxed.
        If the episode is done, the reward is also the diference of the final money
        from the start money, multiplied by a weight.

        Note:
            The reward is divided by a value of 100 to avoid very small learning rates.

        Returns:
            The reward for the step.
        """
        end_w = 10
        daily_w = 1
        sell_w = 1
        reward = 0
        if self._check_done():
                reward += end_w*(self._full_value() - self.start_money)
        # reward += daily_w*self._daily_returns()
        reward += sell_w*self.sold_profit
        reward += self.taxes_reward
        reward /= 100
        self.ep_ret += reward
        return reward

    def _log_step(self, log_all, action, reward, done):
        """Log every step or the episode.

        Args:
            log_all (bool): Log every step if true, otherwise just the end of the episode.
            action (vector): The actions made at the timestep
            reward (float): Ammount of reward recived
            done (bool): Indicate if is the end of the episode
         
        """
        if log_all is True:
            print(f"============ Episode Step {self.ep_step}")
            print(f"Estado Previo: {self.pre_state}")
            print(f"Acoes: {action}")
            print(f"Estado: {self.state}")
            print(f"Shares: {self.shares}")
            print(f"Saldo: {self.balance:.2f}")
            print(f"Preco dos Ativos: {[self._current_price(x) for x in self.shares]}")
            print(f"Lucro Geral: {(self._full_value() - self.start_money):.2f}")
            print(f"Lucro Diario: {self._daily_returns():.2f}")
            print(f"Valor da Carteira: {self._full_value():.2f}")
            print(f"Recompensa: {reward}")
            print(f"Done: {done}")
        if done is True:
            print(f"Lucro Total: {(self._full_value() - self.start_money):.2f}")
            print(f"Overall Reward: {self.ep_ret:.2f}")

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            The initial observation.
        
        """
        self.ep_step = -1
        self.episode += 1
        self.pre_state = []
        self.balance = self.start_money
        self.ep_ret = 0
        self.shares = {}
        for i in range(0, self.n_insiders):
            self.shares[i] = {}

        return self._make_observation()

    def _get_info(self):
        """Get Info.
        
        Needed to overwrite to work with gym. For more information check 
        the gym documentation.

        """
        return {}

    def render(self, mode='human', close=False):
        """Renders the environment.
        
        Needed to overwrite to work with gym. For more information check 
        the gym documentation.
        
        """
        self._log_step()