from utils.logging import set_up_comet_ml, log_to_screen
from utils.utils import (
    update_configs,
    get_agent,
    set_random_seed,
    set_device,
    create_environment,
    get_sweep_from_bsuite_id,
)
from agents.a2c import A2C

import argparse
from experiments.agent_trainer import train_agent
from configs.experiment_config import experiment_config


def run_experiment(args):
    rl_agent = get_agent(args.agent)
    set_device()
    set_random_seed(args.seed)

    env_id_list = get_sweep_from_bsuite_id(args.env)
    for env_id in env_id_list:
        if args.comet:
            tags = [args.agent, args.memory, args.seed, env_id]
            logger = set_up_comet_ml(tags=[*tags])
        else:
            logger = None
        env = create_environment(
            agent=args.agent,
            seed=args.seed,
            memory=args.memory,
            env=env_id,
            window_size=args.window,
        )
        action_size = env.action_space.n
        state_size = env.observation_space.shape[1]
        agent = rl_agent(state_size, action_size, memory=args.memory)
        total_episodes = (
            env.bsuite_num_episodes if args.num_eps is None else args.num_eps
        )
        train_agent(agent, env, total_episodes, logger)


from environment.custom_memory import CustomMemoryChain
from bsuite.utils import gym_wrapper
from bsuite.logging.csv_logging import Logger
from bsuite.utils import wrappers
import termcolor
import random


def test_custom_env():
    env = CustomMemoryChain(memory_length=10, num_bits=3, seed=100)
    results_dir = "results/random/none/"
    logger = Logger(bsuite_id="custom", results_dir=results_dir, overwrite=True)
    termcolor.cprint(
        f"Logging results to CSV file for each bsuite_id in {results_dir}.",
        color="yellow",
        attrs=["bold"],
    )
    env = wrappers.Logging(env, logger, log_by_step=False)
    env = gym_wrapper.GymFromDMEnv(env)

    # for eps in range(10000):
    #     state = env.reset()
    #     while True:
    #         state, reward, done, _ = env.step(random.randrange(env.action_space.n))
    #         if done:
    #             break
    #     if eps % 100 == 0:
    #         print(f"Episode: {eps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="transformers-for-rl")
    parser.add_argument("--name", type=str, default="Test")
    parser.add_argument("--agent", type=str)
    parser.add_argument("--memory", type=str, default=None)
    parser.add_argument("--num_eps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--tags", nargs="*", help="Additional comet experiment tags.")
    args = parser.parse_args()

    update_configs(args)
    # run_experiment(args)
    test_custom_env()


if __name__ == "__main__":
    main()
