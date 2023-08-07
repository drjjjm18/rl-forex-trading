from typing import Tuple, List
import random
import numpy as np
import pickle

from gym import Env
from gym.spaces import Discrete, Box

    

class TradeEnvironment(Env):

    def __init__(self, df, initial_balance=1000, slippage=0.001, log=False):
        self.df = df
        self.current_step = 0
        self.max_steps = len(df) - 1

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.units_held = 0
        self.prev_units_held = 0
        self.slippage = slippage
        self.rewards = []
        self.portfolio = []
        self.columns = list(self.df.columns)
        if 'Close' in self.columns:
            self.columns.remove('Close')
        self.previous_value = self.current_value
        self.log = log

        self.action_space = Discrete(3)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(self.columns)+2,), dtype=np.float32)


    def reset(self):
        self.current_step = 0
        self.units_held = 0
        self.prev_units_held = 0
        self.balance = self.initial_balance
        self.previous_value = self.current_value
        return self._get_observation()

    def step(self, action):
        if self.current_step >= self.max_steps:
            with open('rewards.pkl', 'wb') as f:
                pickle.dump(self.rewards, f)
            with open('portfolio.pkl', 'wb') as f:
                pickle.dump(self.portfolio, f)
            return self._get_observation(), 0, True, {}
        
        self.current_step += 1
        # Execute the action and get the next observation
        current_price = self.df['Close'].iloc[self.current_step]
        if current_price == 0:
            return self._get_observation(), 0, False, {}

        self._take_action(action, current_price)
        reward = self._get_reward()
        self.rewards.append(reward)
        self.portfolio.append(self.current_value)
        next_observation = self._get_observation()

        # Update the previous units held for the next reward calculation
        self.prev_units_held = self.units_held
        self.previous_value = self.current_value

        return next_observation, reward, False, {}

    def _take_action(self, action, current_price):
        if self.log:
            print(action, current_price)
        if action == 1:  # Buy
            if self.balance <= 0:
                return  # Can't buy with zero or negative balance

            max_units = self.balance // current_price
            units_to_buy = max_units
            cost = (1 + self.slippage) * units_to_buy * current_price

            if self.balance < cost:
                units_to_buy = int(self.balance // ((1 + self.slippage) * current_price))
                cost = (1 + self.slippage) * units_to_buy * current_price

            self.balance -= cost
            self.units_held += units_to_buy
        
        elif action == 2:  # Sell
            if self.units_held <= 0:
                return  # Can't sell with zero or negative units_held

            units_to_sell = self.units_held
            revenue = (1 - self.slippage) * units_to_sell * current_price

            self.balance += revenue
            self.units_held -= units_to_sell
        if self.log:
            print(f'balance: {self.balance}, units held: {self.units_held}, value: {self.current_value}')

    def _get_observation(self):
        # Get the current observation (next row of data) from the DataFrame
        observation = self.df.iloc[self.current_step][self.columns].values.astype(np.float32)
        # Append the balance and units held to the observation
        balance = np.array([self.balance], dtype=np.float32)
        units_held = np.array([self.units_held], dtype=np.float32)
        observation = np.concatenate([observation, balance, units_held])
        return observation

    def _get_reward(self):
        # Calculate the portfolio value at the current timestep
        portfolio_value = self.current_value

        # Calculate the percentage change in portfolio value from the previous timestep
        prev_portfolio_value = self.previous_value
        reward = (portfolio_value - prev_portfolio_value) / prev_portfolio_value * 100.0
        if self.log:
            print(f'portfolio value: {portfolio_value}, prev value: {prev_portfolio_value}, reward: {reward}')
        return reward 
    
    @property
    def current_value(self):
        return self.balance + (self.units_held * self.df['Close'].iloc[self.current_step])
