import comet_ml
from utils.logging import set_up_comet_ml
from utils.utils import update_configs_from_args, model_from_args, set_random_seed, get_device, create_environment
from algorithms.a2c import A2C
import argparse
from bsuite import sweep

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="transformers-for-rl")
parser.add_argument("--name", type=str, default="Test")
parser.add_argument("--algo", type=str, default="A2C")
parser.add_argument("--lstm", action="store_true")
parser.add_argument("--transformer", type=str, default=None)
parser.add_argument("--num_eps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--env", type=str)
parser.add_argument("--window", type=int, default=1)
parser.add_argument("--comet", action="store_true")
parser.add_argument("--tags", nargs="*",
                    help="Additional comet experiment tags.")
args = parser.parse_args()


def run_experiment(name, logger: None, algo, model, total_episodes, seed):
    device = get_device()
    set_random_seed(seed)

    if args.env == "all":
        for env in sweep.SWEEP:
            env = create_environment(env)
            rl_head = algo(name, model, env, device, logger)
            rl_head.learn(total_episodes=total_episodes,
                          window_size=args.window)
    else:
        env = create_environment()
        rl_head = algo(name, model, env, device, logger)
        rl_head.learn(total_episodes=total_episodes, window_size=args.window)

    # BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
    # BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
    # __radar_fig__ = summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)


if __name__ == "__main__":
    update_configs_from_args(args)
    if args.comet:
        logger = set_up_comet_ml(
            tags=[args.algo, args.transformer, args.seed, args.env, args.tags]
        )
    else:
        logger = None

    # algo = algo_from_string(args.algo.lower())
    algo = A2C
    model = model_from_args(args)

    run_experiment(args.name, logger, algo, model, args.num_eps, args.seed)
