import torch 
import numpy as np
from algorithms.ppo import train
from models.transformer_actor_critic import TransformerActorCritic
from configs.ppo_config import ppo_config 
from configs.env_config import env_config
from ray import tune


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = {"actor_critic_model": TransformerActorCritic, **ppo_config}
    train(epochs=5000, steps_per_epoch=1000, repeat_action=4, seed=seed, ppo_config=config, env_config=env_config)


if __name__ == '__main__':
    main()