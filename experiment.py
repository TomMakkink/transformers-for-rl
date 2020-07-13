import comet_ml
from algorithms.ppo import PPO
from experiments.experiment_runner import run_experiment
from configs.env_config import env_config
from configs.experiment_config import experiment_config
from configs.transformer_config import transformer_config
from utils.logging import set_up_comet_ml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default="transformers-for-rl")
parser.add_argument('--name', type=str, default="Test")
parser.add_argument('--algo', type=str, default="PPO")
parser.add_argument('--transformer', type=str, default='vanilla')
parser.add_argument('--t_steps', type=int, default=100000)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--env', type=str, default='cartpole/0')
args = parser.parse_args()

if __name__ == '__main__':
    # Boiler Plate to update configs based on input argmuments
    if args.project: experiment_config.update({"project_name": args.project})
    if args.name: experiment_config.update({"experiment_name": args.name})
    if args.transformer: transformer_config.update({"transformer_type": args.transformer})
    if args.env: env_config.update({"env_name": args.env})

    # logger = set_up_comet_ml(tags=[args.algo, args.transformer, args.seed, args.env])
    logger = None
    run_experiment(args.name, logger, total_timesteps=args.t_steps, seed=args.seed) 



