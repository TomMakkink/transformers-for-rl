import numpy as np
import torch

from examples.pendulum import main

import gym

from gym.wrappers import FrameStack

if __name__ == '__main__':
    main("Pendulum/PPO2_Test", total_timesteps=500000, seed=10)
