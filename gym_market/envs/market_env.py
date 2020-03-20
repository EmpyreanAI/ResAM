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
        self.money = start_money
        self.n_insiders = n_insiders
        self.assets_prices = assets_prices
        self.insiders_preds = insiders_preds
        self.sold_profit = 0
        self.num_bought = 0
        self.ep_step = -1
        self.assets = []
        self.oversell = 0
        self.oversbuy = 0
        self.episode = 0
        for i in range(0, n_insiders):
            self.assets.append(deque())

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

    def _portfolio_value(self):
        value = 0
        for i, asset in enumerate(self.assets):
            value += len(asset)*self._day_price(i)
        return value

    def _full_value(self):
        value = self._portfolio_value()
        value += self.money
        return value

    def _day_price(self, asset):
        return self.assets_prices[asset][self.ep_step]

    def _observation_space(self):
        # wallet money, portifolio value
        money_low = [0, 0, 0, 0, 0]
        money_high = [np.inf, np.inf, 1, 1, 1]
        money = spaces.Box(np.array(money_low), np.array(money_high))
        # 0 ou 1 para cada insider
        # insiders_preds = spaces.MultiBinary(self.n_insiders)
        # return spaces.Tuple((money, insiders_preds))
        return money

    def _action_space(self):
        """For each of the stocks.

        - Buy 0 to -inf (e.g. 1.23 = 123% of the actual ammount)
        - Sell 0 to 100
        """
        action_low = [-1, -1, -1]
        action_high = [1, 1, 1]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def step(self, action):
        """.

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.

        """
        self.ep_step += 1
        scope_actions = list(action)
        self._take_action(scope_actions)
        self.state = self._make_observation()
        reward = self._get_reward()
        done = self._check_done()
        info = self._get_info()
        if done is True:
            print(self._full_value() - self.start_money)
        self.pre_state = self.state
        return self.state, reward, done, info

    def _check_done(self):
        bankrupcy = self._full_value() < 0.8*self.start_money
        data_end = len(self.assets_prices[0])-1 == self.ep_step
        if bankrupcy or data_end:
            return True
        else:
            return False

    def _calc_appreciation(self):
        apretiation = 0
        for i, asset in enumerate(self.assets):
            actual_price = self._day_price(i)
            for bought_price in asset:
                apretiation += (actual_price - bought_price)
        return apretiation

    def _sell_action(self, scope_actions):
        for i, action in enumerate(scope_actions):
            if action > 0:
                actual_price = self._day_price(i)
                num_sold = 0
                to_sell = int(len(self.assets[i]) * action)
                while num_sold < to_sell and len(self.assets[i]) > 0:
                    bought_price = self.assets[i].popleft()
                    self.sold_profit += actual_price - bought_price
                    num_sold += 1
                self.money += (actual_price*to_sell)
                scope_actions[i] = 0.0

    def _buy_action(self, scope_actions):
        total = sum(scope_actions)
        if total != 0:
            normal_actions = [x/abs(total) for x in scope_actions]
            for i, action in enumerate(normal_actions):
                action = abs(action)
                actual_price = self._day_price(i)
                to_buy = int(int(action*self.money) // actual_price)
                self.assets[i].extend([actual_price]*to_buy)
                self.num_bought = to_buy
                self.money -= (actual_price*to_buy)


    def _take_action(self, scope_actions):
        self.sold_profit = 0
        self.num_bought = 0
        self._sell_action(scope_actions)
        self._buy_action(scope_actions)



    def _make_observation(self):
        obs = np.zeros(self.observation_space.shape)
        obs[0] = self.money  # wallet money
        obs[1] = self._portfolio_value()  # portifolio money
        for i in range(2, self.n_insiders+2):
            obs[i] = self.insiders_preds[i-2][self.ep_step]

        return obs

    def _get_reward(self):
        """Reward is given for XY."""
        buy_w = -1
        end_w = 1
        reward = 0
        if self._check_done():
            reward += end_w*(self._full_value() - self.start_money)
        reward += (self.num_bought*buy_w)
        reward += self._calc_appreciation()
        reward += self.sold_profit
        return reward

    def _get_info(self):
        return {}

    def render(self, mode='human', close=False):
        self._log_step()

    def _log_step(self):
        print("\nStep {}".format(self.ep_step))
        print("\tMoney {}".format(self.money))
        for i, asset in enumerate(self.assets):
            print(f"\tASSET {i}:")
            print("\t\tNum Assets: {}".format(len(asset)))
            print(f"\t\tAsset Price: {self.assets_prices[i][self.ep_step]}")

    def reset(self):
        self.ep_step = -1
        self.episode += 1
        self.money = self.start_money
        self.assets = []
        for i in range(0, self.n_insiders):
            self.assets.append(deque())
        return self._make_observation()

    # def seed(self, seed):
    #     """Seed."""
    #     random.seed(seed)
    #     np.random.seed


if __name__ == "__main__":
    env = MarketEnv(3, 20, [[1, 2, 3],[3, 2, 1],[2, 3, 1]],
                    [[1, 1, 0],[0, 0, 0 ],[1, 0, 0]])
    env.ep_step += 1
    env._take_action([0.7, -0.3, -0.1])
