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
            tags = [
                args.agent,
                args.memory,
                f"seed={args.seed}",
                env_id,
                f"window={args.window}",
            ]
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


from environment.environment_wrapper import SlidingWindowEnv
from bsuite.utils import gym_wrapper
import random


def test_custom_env(total_episodes):
    env = CustomMemoryChain(memory_length=5, num_bits=3)
    env = gym_wrapper.GymFromDMEnv(env)
    env = SlidingWindowEnv(env, window_size=1)

    for episode in range(total_episodes):
        state = env.reset()
        t = 0
        print(f"Timestep = {t} \n\tState: {state}")
        while True:
            action = random.randint(0, env.action_space.n - 1)
            state, reward, done, _ = env.step(action)
            t += 1
            print(
                f"Timestep = {t} \n\tState: {state} \n\tReward: {reward} \n\tAction: {action}"
            )
            if done:
                break


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
    run_experiment(args)
    # test_custom_env(1)


if __name__ == "__main__":
    main()
