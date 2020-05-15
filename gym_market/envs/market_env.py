"""."""

import gym
import random
import numpy as np
from gym import spaces
from collections import deque


class MarketEnv(gym.Env):
    """."""

    metadata = {'render.modes': ['human']}

    def __init__(self, n_insiders, start_money, assets_prices, insiders_preds):
        """Constructor."""
        self.start_money = start_money
        self.balance = start_money
        self.n_insiders = n_insiders
        self.assets_prices = assets_prices
        self.insiders_preds = insiders_preds

        self.episode = 0
        self.ep_step = -1
        self.ep_ret = 0


        self.pre_state = []
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.taxes = 0
        self.min_allotment = 100
        self.shares = {}
        for i in range(0, n_insiders):
            self.shares[i] = {}

    def _observation_space(self):
        money_low = [0, 0, 0, 0, 0]
        money_high = [np.inf, np.inf, 1, 1, 1]
        money = spaces.Box(np.array(money_low), np.array(money_high))
        return money

    def _action_space(self):
        """For each of the stocks.

        - Buy 0 to -inf (e.g. 1.23 = 123% of the actual ammount)
        - Sell 0 to 100
        """
        action_low = [-1, -1, -1]
        action_high = [1, 1, 1]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _daily_returns(self):
        day_ret = 0
        for share_index in self.shares:
            share = self.shares[share_index]
            day_price = self._current_price(share_index)
            for holding in share:
                holding_amt = share[holding]
                day_ret += (day_price - holding)*holding_amt
        return day_ret

    def _portfolio_value(self):
        value = 0
        for share_index in self.shares:
            share = self.shares[share_index]
            for holding in share:
                value += share[holding]*self._current_price(share_index)
        return value

    def _full_value(self):
        value = self._portfolio_value()
        value += self.balance
        return value

    def _current_price(self, asset):
        return self.assets_prices[asset][self.ep_step]

    def step(self, action):
        self.ep_step += 1
        action_backup = list(action)
        self._take_action(action)
        self.state = self._make_observation()
        reward = self._get_reward()
        done = self._check_done()
        info = self._get_info()
        # self._pls_help(action_backup, reward, done)
        if done is True:
            print(f"Lucro Total: {(self._full_value() - self.start_money):.2f}")
            print(f"Overall Reward: {self.ep_ret:.2f}")
        self.pre_state = self.state
        return self.state, reward, done, info

    def _check_done(self):
        bankrupcy = self._full_value() < 0.8*self.start_money
        data_end = len(self.assets_prices[0])-1 == self.ep_step
        if bankrupcy or data_end:
            return True
        else:
            return False

    def _sell_action(self, actions):
        # sold_profit = 0
        num_sold = 0
        for share_index, action in enumerate(actions):
            if action > 0.5:
                day_price = self._current_price(share_index)
                share = self.shares[share_index]
                for holding in share:
                    # sold_profit += (day_price - holding)*share[holding]
                    num_sold += share[holding]
                self.balance += (day_price*num_sold)
                share.clear()

    def _buy_action(self, actions):
        total = abs(sum(actions))
        if total > 0:
            money = self.balance
            actions_normal = [abs(x) for x in actions]
            if total > 1:
                actions_normal = [x/total for x in actions]
            for share_index, action in enumerate(actions_normal):
                if action > 0:
                    day_price = self._current_price(share_index)

                    to_buy = int(action*money // day_price)

                    if day_price in self.shares[share_index]:
                        self.shares[share_index][day_price] += to_buy
                    else:
                        self.shares[share_index][day_price] = to_buy

                    self.balance -= (day_price*to_buy)

    def _take_action(self, actions):
        self._sell_action(actions)
        actions = [0 if x > 0 else x for x in actions]
        self._buy_action(actions)

    def _make_observation(self):
        obs = np.zeros(self.observation_space.shape)
        obs[0] = self.balance
        obs[1] = self._portfolio_value()
        for i in range(2, self.n_insiders+2):
            obs[i] = self.insiders_preds[i-2][self.ep_step]

        return obs

    def _get_reward(self):
        """Reward is given for XY."""
        end_w = 5
        reward = 0
        if self._check_done():
            reward += end_w*(self._full_value() - self.start_money)
        reward += 0.5*self._daily_returns()
        self.ep_ret += reward
        return reward/10

    def _pls_help(self, action, reward, done):
        print(f"============ Episode Step {self.ep_step}")
        print(f"Estado Previo: {self.pre_state}")
        print(f"Acoes: {action}")
        print(f"Estado: {self.state}")
        print(f"Saldo: {self.balance:.2f}")
        print(f"Lucro Geral: {(self._full_value() - self.start_money):.2f}")
        print(f"Lucro Diario: {self._daily_returns():.2f}")
        print(f"Recompensa: {reward}")
        print(f"Done: {done}")


    def _log_step(self):
        print("\nStep {}".format(self.ep_step))
        print("\tMoney {}".format(self.balance))
        for i, asset in enumerate(self.assets):
            print(f"\tASSET {i}:")
            print("\t\tNum Assets: {}".format(len(asset)))
            print(f"\t\tAsset Price: {self.assets_prices[i][self.ep_step]}")

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
    market = MarketEnv(3, 1000, [[12.3, 13.1, 14.1], [14.2, 13.3, 12.4], [23.2, 12.2, 13.1]],
                       [[1, 1, 0], [0, 0, 0], [0, 1, 0]])
    market.ep_step = 0
    market._take_action([1, -.32, -.642])
    # market._buy_action([0, 0.5, 0.5])
    # print(market.shares)
    # market.ep_step = 1
    # print(market.money)
    # ret = market._swing_trade_return()
    # print(ret)
