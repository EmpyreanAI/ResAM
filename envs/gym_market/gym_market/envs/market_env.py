"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import gym
import random
import numpy as np
from gym import spaces
from collections import deque

class MarketEnv(gym.Env):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, n_insiders, start_money, assets_prices, insiders_preds):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
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
            A spaces.Box() value with the observation space
        """
        money_low = [0, 0, 0, 0, 0]
        money_high = [np.inf, np.inf, 1, 1, 1]
        money = spaces.Box(np.array(money_low), np.array(money_high))
        return money

    def _action_space(self):
        """Define the action space of the environment.

        The current selected action space is a vector of values between -1 and 1 where:
            - [-1, 0) = Buy x percentage of the available resources in this stock
            - 0 = Do noting
            - 1 = Sell Everything

        Returns:
            A spaces.Box() value with the action space
        """
        action_low = [-1, -1, -1]
        action_high = [1, 1, 1]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _daily_returns(self):
        """.

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
            asset (int): Index of the asset in the vector to get the price

        Returns:
            The current price of the asset at a time-step.

        """
        return self.assets_prices[asset][self.ep_step]

    def step(self, action):
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
        self.taxes_reward = 0
        self._sell_action(actions)
        actions_normal = self._normalize(actions)
        self._buy_action(actions_normal)

    def _make_observation(self):
        obs = np.zeros(self.observation_space.shape)
        obs[0] = self.balance
        obs[1] = self._portfolio_value()
        obs[2] = self.insiders_preds[0][self.ep_step]
        obs[3] = self.insiders_preds[1][self.ep_step]
        obs[4] = self.insiders_preds[2][self.ep_step]

        return obs

    def _get_reward(self):
        """Reward is given for XY."""
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

    # def _log_buy(self, log, info):
    #         print(f"Share {share_index}: Price {day_price}")
    #         print(f"Allotment Price: {allotment_price}")
    #         print(f"Avaiable Money: {alloc_money}, Percentage: {action}")
    #         print(f"Allotment Ammount: {allot_buy_amt}")
    #         print(f"Total Shares: {total_shares}")
    #         print(f"Total Costs: {total_shares*day_price}\n\n")


    def _log_step(self, log_all, action, reward, done):
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
        return {}

    def render(self, mode='human', close=False):
        self._log_step()


if __name__ == "__main__":
    market = MarketEnv(3, 10000, [[12.3, 13.1, 14.1], [14.2, 13.3, 12.4],
                                  [23.2, 12.2, 13.1]],
                       [[1, 1, 0], [0, 0, 0], [0, 1, 0]])
    market.ep_step = 0
    market.step([0.7, -.7, -.6])
    # market._take_action([1, -.7, -.6])
    # print(market.shares)
    # print(market.balance)
