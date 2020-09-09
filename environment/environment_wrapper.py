from gym import Wrapper
from collections import deque
from configs.experiment_config import experiment_config
import torch


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
        self.pad_obs_window(self.obs_window)
        state = self.env.reset(**kwargs)
        state = state.squeeze(0)
        state = torch.from_numpy(state).float().to(self.device)
        self.obs_window.append(state)
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

    def pad_obs_window(self, obs_window):
        for i in range(obs_window.maxlen - 1):
            obs_window.append(torch.zeros(self.env.observation_space.shape[1]))
