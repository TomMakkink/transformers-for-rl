from algorithms.ppo2 import PPO2
from configs import env_config, ppo_config
from env.cartpole import make_env
from utils.utils import set_random_seed, get_device
from models.mlp_actor_critic import MLPActorCritic


def main(name, epochs, seed):
        device = get_device()
        set_random_seed(seed)

        env = make_env(**env_config, seed=seed, device=device)
        model = PPO2(**ppo_config, env=env, seed=seed, device=device)

        model.learn(epochs)
    
        
