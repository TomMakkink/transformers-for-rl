import bsuite
from bsuite.utils import gym_wrapper

from algorithms.ppo import PPO
from configs.ppo_config import ppo_config
<<<<<<< HEAD
from utils.utils import set_random_seed, get_device, process_obs
=======
>>>>>>> f037699e9a0ce91bed72bebfa5c45684f80f87bb
from models.mlp_cat_actor_critic import MLPActorCritic
import bsuite
from bsuite.utils import gym_wrapper
from gym.wrappers import TransformObservation
<<<<<<< HEAD
import torch
=======
from utils.utils import set_random_seed, get_device
>>>>>>> f037699e9a0ce91bed72bebfa5c45684f80f87bb

# Cart pole environment
#   The observation is a vector representing:
#     `(x, x_dot, sin(theta), cos(theta), theta_dot, time_elapsed)
#      (position, velocity, sin(angle), cos(angle), angular velocity, time_elapsed)
#   The actions are discrete ['left', 'stay', 'right']. Episodes start with the
#   pole close to upright. Episodes end when the pole falls, the cart falls off
#   the table, or the max_time is reached.

def cartpole_test(name, experiment, total_timesteps, seed):
        device = get_device()
        set_random_seed(seed)
        raw_env = bsuite.load_from_id(bsuite_id='cartpole/0')
        env = gym_wrapper.GymFromDMEnv(raw_env)
        # Transform observations from [1, 6] -> [6]
<<<<<<< HEAD
        env = TransformObservation(env, lambda obs: obs.squeeze()) #process_obs(obs, device))
        model = PPO(**ppo_config, name=name, actor_critic=MLPActorCritic, env=env, device=device)
        model.learn(total_timesteps)
=======
        env = TransformObservation(env, lambda obs: obs.squeeze())
        experiment.log_parameters(ppo_config)
        model = PPO(**ppo_config, name=name, actor_critic=MLPActorCritic, env=env, device=device, experiment=experiment)
        model.learn(total_timesteps)
>>>>>>> f037699e9a0ce91bed72bebfa5c45684f80f87bb
