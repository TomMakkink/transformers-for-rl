from gym import Wrapper
from collections import deque
from configs.experiment_config import experiment_config
import torch


def pad_obs_window(obs_window, state):
    for i in range(obs_window.maxlen):
        obs_window.append(state)


class SlidingWindowEnv(Wrapper):
    def __init__(self, env, window_size):
        super(SlidingWindowEnv, self).__init__(env)
        self.obs_window = deque(maxlen=window_size)
        self.device = experiment_config["device"]

    def reset(self, **kwargs):
        """
        Returns: 
            state of shape: (window_size, features)
        """
        state = self.env.reset(**kwargs)
        state = state.squeeze(0)
        state = torch.from_numpy(state).float().to(self.device)
        pad_obs_window(self.obs_window, state)
        state = torch.stack(list(self.obs_window))
        return state

    def step(self, action):
        """
        Returns: 
            state of shape: (window_size, features)
        """
        state, reward, done, info = self.env.step(action)
        state = state.squeeze(0)
        state = torch.from_numpy(state).float().to(self.device)
        self.obs_window.append(state)
        state = torch.stack(list(self.obs_window))

        return (state, reward, done, info)
