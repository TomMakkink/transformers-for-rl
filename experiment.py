import numpy as np
import torch

from examples.pendulum import main


if __name__ == '__main__':
    train(name="pendulum/ppo_test", epochs=100, seed=seed)
