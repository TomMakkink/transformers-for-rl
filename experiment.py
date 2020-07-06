import comet_ml

from experiments.cartpole import cartpole_test

# from experiments.ppo_test import ppo_test
from experiments.transformer_cartpole import transformer_cartpole
from configs.transformer_config import transformer_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="Test")
parser.add_argument('--transformer', type=str, default='vanilla')
parser.add_argument('--t_steps', type=int, default=500000)
parser.add_argument('--seed', type=int, default=10)

args = parser.parse_args()

if __name__ == '__main__':
<<<<<<< HEAD
    cartpole_test(args.name, total_timesteps=5000, seed=10)
=======
    experiment = comet_ml.Experiment(project_name="transformers-for-rl", log_code=False,
                                     log_git_metadata=False, log_git_patch=False, log_env_host=False)
    experiment.add_tag(args.name)
    experiment.add_tag(args.transformer)
    experiment.add_tag(args.seed)

    cartpole_test(args.name, experiment, total_timesteps=500000, seed=10)

>>>>>>> f037699e9a0ce91bed72bebfa5c45684f80f87bb
    # transformer_config.update({"transformer_type": args.transformer})
    # transformer_cartpole(args.name, total_timesteps=args.t_steps, seeds=args.seed) 

    # transformer_config.update({"transformer_type": "vanilla"})
    # transformer_cartpole("CartPole/PPO/Canoncial", total_timesteps=50000, seed=10)

    # transformer_config.update({"transformer_type": "gtrxl"})
    # transformer_cartpole("CartPole/PPO/GTrXL", total_timesteps=50000, seed=10)

    # transformer_config.update({"transformer_type": "rezero"})
    # transformer_cartpole("CartPole/PPO/ReZero", total_timesteps=50000, seed=10)
