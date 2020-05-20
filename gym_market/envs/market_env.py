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
        self.sold_profit = 0

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
        money_low = [0, 0, 0, 0, 0, 0]
        money_high = [1, np.inf, 1, np.inf, 1, np.inf]
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
        action = [1 if x > .5 else 0 if x > 0 else x for x in action]
        self._take_action(action)
        self.state = self._make_observation()
        reward = self._get_reward()
        done = self._check_done()
        info = self._get_info()
        # self._log_step(False, action, reward, done)
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
        # obs[0] = self.balance + self._full_value()
        obs[0] = self.insiders_preds[0][self.ep_step]
        obs[1] = self._current_price(0)
        obs[2] = self.insiders_preds[1][self.ep_step]
        obs[3] = self._current_price(1)
        obs[4] = self.insiders_preds[2][self.ep_step]
        obs[5] = self._current_price(2)

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
