from utils.utils import set_random_seed, get_device, create_environment
from models.transformer_categorical_actor_critic import TransformerActorCritic
from algorithms.ppo import PPO

def run_experiment(name, logger: None, algo=PPO, 
                   model=TransformerActorCritic, total_timesteps=50000, seed=10):
    device = get_device()
    set_random_seed(seed)
    env = create_environment()
    if logger: logger.log_parameters(ppo_config)
    model = algo(name, model, env, device, logger)
    model.learn(total_timesteps)
