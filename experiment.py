import comet_ml

from experiments.cartpole import cartpole_test

# from experiments.ppo_test import ppo_test
from experiments.transformer_cartpole import transformer_cartpole
from configs.transformer_config import transformer_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="Test")
parser.add_argument('--transformer', type=str, default='vanilla')
parser.add_argument('--t_steps', type=int, default=100000)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--environment', type=str, default='cartpole/0')

args = parser.parse_args()

if __name__ == '__main__':
    experiment = comet_ml.Experiment(project_name="transformers-for-rl", log_code=False,
                                     log_git_metadata=False, log_git_patch=False, log_env_host=False)
    experiment.set_name(args.name)
    experiment.add_tag(args.seed)
    experiment.add_tag(args.environment)
    # experiment.add_tag(args.name)
    experiment.add_tag(args.transformer)

    transformer_config.update({"transformer_type": args.transformer})
    transformer_cartpole(args.name, experiment, env=args.environment, total_timesteps=args.t_steps, seed=args.seed) 
