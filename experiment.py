import torch 
import numpy as np
from examples.cartpole_ppo import train
from env.cartpole import make_env
from models.mlp_categorical_actor_critic import MLPActorCritic
# from examples.cart_pole_ppo import ppo
# from examples.cart_pole_single_loss import ppo
from models.transformer_actor_critic import TransformerActorCritic
from configs.ppo_config import ppo_config 
from configs.env_config import env_config


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # config = {"actor_critic_model": TransformerActorCritic, **ppo_config}
    # train(epochs=5000, steps_per_epoch=1000, repeat_action=4, seed=seed, ppo_config=config, env_config=env_config)
    # ppo()
    train(env_fn=make_env, actor_critic=MLPActorCritic, seed=seed)


if __name__ == '__main__':
    main()