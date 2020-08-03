from utils.utils import set_random_seed, get_device, create_environment
from models.transformer_categorical_actor_critic import TransformerActorCritic
from algorithms.a2c import A2C
from models.transformer_a2c import TransformerA2C

from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis

from bsuite import sweep
sweep.SWEEP


def run_experiment(
    name, logger: None, algo=A2C, model=TransformerA2C, total_episodes=1000, seed=10
):
    device = get_device()
    set_random_seed(seed)
    env = create_environment()
    model = algo(name, model, env, device, logger)
    model.learn(total_episodes=total_episodes, window_size=1)

    # DF, _ = csv_load.load_bsuite("results/")
    # BSUITE_SCORE = summary_analysis.bsuite_score(DF)
    # summary_analysis.plot_single_experiment(BSUITE_SCORE, "memory_len")
