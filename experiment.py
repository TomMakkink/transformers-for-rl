from utils.logging import set_up_comet_ml, log_to_screen
from utils.utils import (
    update_configs,
    get_agent,
    set_random_seed,
    set_device,
    create_environment,
)
from agents.a2c import A2C

import argparse
from experiments.agent_trainer import train_agent
from configs.experiment_config import experiment_config


def run_experiment(args):
    agent = get_agent(args.agent)
    set_device()
    set_random_seed(args.seed)

    if args.comet:
        tags = [args.agent, args.memory, args.seed, args.env]
        logger = set_up_comet_ml(tags=[*tags])
    else:
        logger = None
    env = create_environment(
        agent=args.agent, seed=args.seed, memory=args.memory, window_size=args.window
    )
    action_size = env.action_space.n
    state_size = env.observation_space.shape[1]
    agent = agent(state_size, action_size, hidden_size=[128], memory=args.memory)
    train_agent(agent, env, args.num_eps, logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="transformers-for-rl")
    parser.add_argument("--name", type=str, default="Test")
    parser.add_argument("--agent", type=str)
    parser.add_argument("--memory", type=str, default=None)
    parser.add_argument("--num_eps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--tags", nargs="*", help="Additional comet experiment tags.")
    args = parser.parse_args()

    update_configs(args)
    # log_to_screen(experiment_config)
    run_experiment(args)


if __name__ == "__main__":
    main()
