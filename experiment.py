import torch 
import numpy as np
from algorithms.ppo import ppo 
from models.mlp_actor_critic import MLPActorCritic
from models.transformer_actor_critic import TransformerActorCritic
from models.conv_actor_critic import ConvActorCritic
from configs.ppo_config import ppo_config 


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ppo(env_name= "CarRacing-v0", actor_critic=ConvActorCritic, seed=seed, **ppo_config)
    ppo(env_name= "CarRacing-v0", actor_critic=TransformerActorCritic, seed=seed, **ppo_config)

if __name__ == '__main__':
    main()