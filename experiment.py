import torch 
import numpy as np
from env.cartpole import make_env
from utils.utils import set_random_seed, get_device 
from models.mlp_actor_critic import MLPActorCritic
from examples.pendulum import train
# from examples.ppo_test import train

def main():
    device = get_device()
    seed = 42
    set_random_seed(seed)

    train(name="pendulum/ppo_test", env_fn=make_env, steps_per_epoch=2048, actor_critic=MLPActorCritic, seed=seed, device=device)


if __name__ == '__main__':
    main()