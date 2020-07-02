from algorithms.ppo import PPO
from configs.ppo_config import ppo_config
from utils.utils import set_random_seed, get_device
from models.mlp_cat_actor_critic import MLPActorCritic
import bsuite
from bsuite.utils import gym_wrapper

# Cart pole environment
#   The observation is a vector representing:
#     `(x, x_dot, sin(theta), cos(theta), theta_dot, time_elapsed)
#      (position, velocity, sin(angle), cos(angle), angular velocity, time_elapsed)
#   The actions are discrete ['left', 'stay', 'right']. Episodes start with the
#   pole close to upright. Episodes end when the pole falls, the cart falls off
#   the table, or the max_time is reached.

def main(name, total_timesteps, seed):
        device = get_device()
        set_random_seed(seed)
        raw_env = bsuite.load_from_id(bsuite_id='cartpole/0')
        env = gym_wrapper.GymFromDMEnv(raw_env)
        model = PPO(**ppo_config, name=name, actor_critic=MLPActorCritic, env=env, device=device)
        model.learn(total_timesteps)