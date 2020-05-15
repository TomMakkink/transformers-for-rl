import gym 
from gym.wrappers import FrameStack, TransformObservation
import numpy as np
import torch

def make_env(env_name='CartPole-v0', max_episode_steps=500, frame_stack=4, seed=0, device="cpu"):
    env = gym.make(env_name)
    env._max_episode_steps = max_episode_steps
    if frame_stack > 1: 
        env = FrameStack(env, frame_stack)
        env = TransformObservation(env, lambda obs: _process_stacked_obs(obs, device))
    else:
        env = TransformObservation(env, lambda obs: _process_obs(obs, device))
    env.seed(seed)
    return env

def _process_stacked_obs(obs, device):
    obs = np.array(obs, copy=False)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return obs


def _process_obs(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device)

def _flatten_stacked_obs(obs, device):
    obs = np.array(obs, copy=False)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return obs.view(-1)