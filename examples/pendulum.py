from algorithms.ppo2 import PPO2
from configs.env_config import env_config
from configs.ppo_config import ppo_config
from env.cartpole import make_env
from utils.utils import set_random_seed, get_device
from models.mlp_actor_critic import MLPActorCritic
# from models.mlp_cat_actor_critic import MLPActorCritic
import gym


def main(name, epochs, seed):
        device = get_device()
        set_random_seed(seed)

        # env = make_env(**env_config, seed=seed, device=device)
        env = gym.make('MountainCarContinuous-v0')
        model = PPO2(**ppo_config, name=name, actor_critic=MLPActorCritic, env=env, device=device)

        model.learn(epochs)
    
        
