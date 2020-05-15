import torch 
import numpy as np
from examples.cartpole_ppo_single_loss import train
from env.cartpole import make_env
# from models.mlp_categorical_actor_critic import MLPActorCritic
from models.transformer_categorical_actor_critic import TransformerActorCritic


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(name="ReZero (8)", env_fn=make_env, actor_critic=TransformerActorCritic, seed=seed, frame_stack=8)


if __name__ == '__main__':
    main()