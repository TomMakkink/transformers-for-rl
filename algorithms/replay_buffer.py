# PPO buffer implementation derived from openai ppo: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from utils.utils import combined_shape, count_vars, discount_cumsum, plot_grad_flow


class ReplayBuffer():
    """
    A buffer for storing complete episodes experienced by a PPO agents, 
    and using Generalizaed Advantage Estimation (GAE-Lambda) to calculate 
    the advantages of state-action pairs. 
    """

    def __init__(self, device, max_epi_num=50, gamma=0.99, lam=0.95):
        self.max_epi_num = max_epi_num
        self.memory = deque(maxlen=max_epi_num)
        # self.memory.append([])
        self.current_epi = -1
        self.is_av = False
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def reset(self):
        self.current_epi = -1
        self.memory.clear()
        # self.memory.append([])

    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def store(self, obs, actions, rewards, values, logp, last_val):
        # The "last_val" argument should be 0 if the trajectory ended
        # because the agent reached a terminal state (died), and otherwise
        # should be V(s_T), the value function estimated for the last state.
        # This allows us to bootstrap the reward-to-go calculation to account
        # for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        # Explicitly convert to numpy array
        actions = np.array(actions, dtype=np.float32)
        logp = np.array(logp, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)

        rewards = np.append(rewards, last_val)
        values = np.append(values, last_val)

        # Call at the end of an episode
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.lam)

        # the next two lines implement the advantage normalization trick (shifted to have
        # mean zero and std one).
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / adv_std

        # the next line computes rewards-to-go, to be targets for the value function
        returns = discount_cumsum(rewards, self.gamma)[:-1]

        self.memory[self.current_epi].append(
            [obs, actions, returns, advantages, logp])

    def get(self, episode_index):
        """
        Call this at the end of an epoch to get all of the data from the buffer. 
        """
        episode = self.memory[episode_index]
        obs, actions, returns, advantages, logp = episode[0][0], episode[
            0][1], episode[0][2], episode[0][3], episode[0][4]

        # Convert from numpy arrays to pytorch tensors
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logp = torch.as_tensor(logp, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(
            returns.copy(), dtype=torch.float32, device=self.device)

        return obs, actions, returns, advantages, logp

    def size(self):
        return len(self.memory)
