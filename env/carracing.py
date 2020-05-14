import numpy as np 
import torch 
import gym
from gym.wrappers import FrameStack, Monitor, ResizeObservation, TransformObservation


def make_env(env_name="CarRacing-v0", max_ep_len=1000, size=64, num_stack=1, seed=43, monitor=True, device="cpu"):
    # TODO: Write this in a better way. 
    env = gym.make(env_name)
    env._max_episode_steps = max_ep_len
    if monitor: env = Monitor(env, './video', force=True)
    if size is not None: env = ResizeObservation(env, size)
    env = FrameStack(env, num_stack=num_stack)
    env = TransformObservation(env, lambda obs: _process_frame(obs, device))
    env.seed(seed)
    return env


def _process_frame(obs, device):
    # Convert LazyFrame to np array. Shape: [Frames, Height, Width, Channels]
    obs = np.array(obs, copy=False)
    # Convert np array to torch tensor. 
    state = torch.as_tensor(obs, dtype=torch.float32, device=device)
    # Convert to channels first format. Shape: [Frame, Channels, Height, Width]
    state = state.permute(0, 3, 1, 2)
    state /= 255
    # Add batch dimension of 1. [batch_size, frame, channels, height, width]
    return state.unsqueeze(0)