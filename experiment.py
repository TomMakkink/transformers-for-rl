import torch 
import numpy as np
from algorithms.ppo import ppo 
from models.transformer_actor_critic import TransformerActorCritic
from configs.ppo_config import ppo_config 
from ray import tune


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = {"env_name": "CarRacing-v0", "actor_critic":TransformerActorCritic, "seed":seed, **ppo_config}
    ppo(**config)
    # tune.run(
    #     ray_ppo, 
    #     config = {"env_name": "CarRacing-v0", "actor_critic":TransformerActorCritic, "seed":seed, **ppo_config},
    #     resources_per_trial = {"cpu": 4, "gpu": 1} ,
    # )
    # ppo(env_name= "CarRacing-v0", actor_critic=ConvActorCritic, seed=seed, **ppo_config)
    # ppo(env_name= "CarRacing-v0", actor_critic=TransformerActorCritic, seed=seed, **ppo_config)

if __name__ == '__main__':
    main()