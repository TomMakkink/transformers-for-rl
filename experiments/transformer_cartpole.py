from algorithms.ppo import PPO
from configs.ppo_config import ppo_config
from utils.utils import set_random_seed, get_device
from models.transformer_categorical_actor_critic import TransformerActorCritic
import bsuite
from bsuite.utils import gym_wrapper
from gym.wrappers import TransformObservation

def transformer_cartpole(name, experiment, total_timesteps, seed):
        device = get_device()
        set_random_seed(seed)
        raw_env = bsuite.load_from_id(bsuite_id='cartpole/0')
        env = gym_wrapper.GymFromDMEnv(raw_env)
        env = TransformObservation(env, lambda obs: obs.squeeze())
        experiment.log_parameters(ppo_config)
        model = PPO(**ppo_config, name=name, actor_critic=TransformerActorCritic, env=env, device=device)
        model.learn(total_timesteps)