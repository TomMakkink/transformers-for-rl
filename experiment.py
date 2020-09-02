from utils.logging import set_up_comet_ml
<<<<<<< HEAD
from utils.utils import update_configs_from_args, model_from_args, set_random_seed, \
    get_device, create_environment, algo_from_string
from algorithms.a2c import A2C
import argparse
from bsuite import sweep
import numpy as np
from models.actor_critic_lstm import ActorCriticLSTM
from models.actor_critic_mlp import ActorCriticMLP
from models.actor_critic_transformer import ActorCriticTransformer

=======
from utils.utils import (
    update_configs_from_args,
    get_agent,
    set_random_seed,
    set_device,
    create_environment,
)
from agents.a2c import A2C
>>>>>>> 66408be54c5c559a731078fee1acad15f67046f4

import argparse
from experiments.agent_trainer import train_agent


def run_experiment(args):
    agent = get_agent(args.agent)
    set_device()
    set_random_seed(args.seed)

<<<<<<< HEAD
    algo = algo_from_string(args.algo)

    model = model_from_args(args)

    device = get_device()
    set_random_seed(seed)
    if args.env == "all":
        for env in sweep.SWEEP:
            logger = get_logger(use_comet, tags, env)
            env = create_environment(alog_name=args.algo, seed=args.seed,
                                     transformer=args.transformer, env=env,
                                     use_lstm=args.lstm)
            rl_head = algo(name, model, env, device, logger)
            rl_head.learn(total_episodes=total_episodes,
                          window_size=args.window)
    else:
        logger = get_logger(use_comet, tags, args.env)
        env = create_environment(
            alog_name=args.algo, seed=args.seed, transformer=args.transformer,
            use_lstm=args.lstm)
        rl_head = algo(name, model, env, device, logger)
        rl_head.learn(
            total_episodes=total_episodes, window_size=args.window)

    # BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
    # BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
    # __radar_fig__ = summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)
=======
    if args.comet:
        tags = [args.agent, args.memory, args.seed, args.env]  # , args.tags]
        logger = set_up_comet_ml(tags=[*tags])
    else:
        logger = None
    env = create_environment(
        agent=args.agent, seed=args.seed, memory=args.memory, window_size=args.window
    )
    action_size = env.action_space.n
    state_size = env.observation_space.shape[1]
    agent = agent(state_size, action_size, args.memory)
    train_agent(agent, env, args.num_eps, logger)
>>>>>>> 66408be54c5c559a731078fee1acad15f67046f4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="transformers-for-rl")
    parser.add_argument("--name", type=str, default="Test")
<<<<<<< HEAD
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--transformer", type=str, default=None)
    parser.add_argument("--num_eps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=10)
=======
    parser.add_argument("--agent", type=str, default="A2C")
    parser.add_argument("--memory", type=str, default=None)
    parser.add_argument("--num_eps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
>>>>>>> 66408be54c5c559a731078fee1acad15f67046f4
    parser.add_argument("--env", type=str)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--tags", nargs="*", help="Additional comet experiment tags.")
    args = parser.parse_args()

    update_configs_from_args(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
