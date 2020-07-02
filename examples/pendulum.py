from algorithms.ppo import PPO
from configs.env_config import env_config
from configs.ppo_config import ppo_config
from env.cartpole import make_env
from utils.utils import set_random_seed, get_device
from models.mlp_actor_critic import MLPActorCritic
import gym


def main(name, total_timesteps, seed):
        device = get_device()
        set_random_seed(seed)
        env = make_env(**env_config, seed=seed, device=device)
        model = PPO(**ppo_config, name=name, actor_critic=MLPActorCritic, env=env, device=device)
        model.learn(total_timesteps)
    
        
