from algorithms.ppo import PPO
from configs.ppo_config import ppo_config
from utils.utils import set_random_seed, get_device, create_environment
from models.transformer_categorical_actor_critic import TransformerActorCritic


def transformer_cartpole(name, experiment, env, total_timesteps, seed):
        device = get_device()
        set_random_seed(seed)
        env = create_environment(env)
        experiment.log_parameters(ppo_config)
        model = PPO(**ppo_config, name=name, actor_critic=TransformerActorCritic, env=env, device=device, experiment=experiment)
        model.learn(total_timesteps)