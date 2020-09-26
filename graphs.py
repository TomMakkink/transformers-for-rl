from itertools import zip_longest
import numpy as np
import os
from os import listdir
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis
from bsuite.experiments.umbrella_distract import analysis as umbrella_distract_analysis


sns.set()

DEFAULT_SAVE_DIR = "graphs"
DEFAULT_DATA_FOLDER = "results"
AGENTS = ["a2c"]
MEMORY = [
    # "mha",
    # "rmha",
    # "gmha",
    "gtrxl",
    # "gtrxl_32_1_1",
    # "gtrxl_64_1_1",
    # "gtrxl_64_2_1",
    # "gtrxl_64_4_1",
    # "gtrxl_64_8_1",
    # "linformer",
    "lstm",
    # "None",
    # "rezero",
    # "vanilla",
    # "xl",
]

ENVS = ["memory_custom"]
ENV_NUMS = list(map(str, range(13)))
COLOURS = ["blue", "green", "red", "purple", "black", "orange"]
WINDOW_SIZES = ["4"]

#   - results
#       - window
#           - agent
#               - memory


import pandas as pd
from environment.custom_memory import score
LEARNING_THRESH = 0.75

def graph_custom_env(root, save):
    colour_index = 0
    experiments = {}
    for window_size in WINDOW_SIZES:
        for model in MEMORY:
            for agent in AGENTS:
                experiments[model] = f"{root_dir}/{window_size}/{agent}/{model}/"

        fig, axs = plt.subplots(1, len(MEMORY), sharey=True)
        x = np.arange(len(ENV_NUMS))
        for i, (name, file_dir) in enumerate(experiments.items()):
            regret = []
            for env_num in ENV_NUMS:
                file = f"{file_dir}bsuite_id_-_memory_custom-{env_num}.csv"
                env_df = pd.read_csv(file)
                env_regret = score(env_df)
                regret.append(env_regret)

            colors = ["blue" if x < LEARNING_THRESH else "red" for x in regret]
            axs[i].scatter(x, regret, s=200, c=colors)
            axs[i].set_title(f"{name}")
            axs[i].set_xlabel("Environments")
            axs[i].set_ylabel("Average Regret at 10000 episodes")
            axs[i].set_xticks(x)

        # y = np.repeat(LEARNING_THRESH, len(ENV_NUMS))
        # ax.plot(x, y, '--')
        # axs.legend()
        plt.savefig(f"{save}/custom_memory_plots.png")
        plt.close()



def bsuite_graphing(root_dir, save_dir):
    for window_size in WINDOW_SIZES:
        experiments = {}
        for model in MEMORY:
            for agent in AGENTS:
                experiments[model] = f"{root_dir}/{window_size}/{agent}/{model}/"

        DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
        BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)

        # Plots specialized to the experiment
        for env in ENVS:
            result = summary_analysis.plot_single_experiment(
                BSUITE_SCORE, env, SWEEP_VARS
            )
            save = save_dir + f"/{window_size}"
            if not os.path.isdir(save):
                os.makedirs(save)
            result.save(f"{save}/{env}_results")

            if env in [
                "memory_len",
                "memory_size",
                "umbrella_length",
                "umbrella_distract",
            ]:
                env_df = DF[DF.bsuite_env == env].copy()
                if env == "memory_len":
                    learning_analysis = memory_len_analysis.plot_learning(
                        env_df, SWEEP_VARS
                    )
                    scale_analysis = memory_len_analysis.plot_scale(env_df, SWEEP_VARS)
                    seeds_analysis = memory_len_analysis.plot_seeds(env_df, SWEEP_VARS)
                elif env == "memory_size":
                    learning_analysis = memory_size_analysis.plot_learning(
                        env_df, SWEEP_VARS
                    )
                    scale_analysis = memory_size_analysis.plot_scale(env_df, SWEEP_VARS)
                    seeds_analysis = memory_size_analysis.plot_seeds(env_df, SWEEP_VARS)
                elif env == "umbrella_length":
                    learning_analysis = umbrella_length_analysis.plot_learning(
                        env_df, SWEEP_VARS
                    )
                    scale_analysis = umbrella_length_analysis.plot_scale(
                        env_df, SWEEP_VARS
                    )
                    seeds_analysis = umbrella_length_analysis.plot_seeds(
                        env_df, SWEEP_VARS
                    )
                elif env == "umbrella_distract":
                    learning_analysis = umbrella_distract_analysis.plot_learning(
                        env_df, SWEEP_VARS
                    )
                    scale_analysis = umbrella_distract_analysis.plot_scale(
                        env_df, SWEEP_VARS
                    )
                    seeds_analysis = umbrella_distract_analysis.plot_seeds(
                        env_df, SWEEP_VARS
                    )

                learning_analysis.save(save_dir + f"/{window_size}/{env}_learning")
                scale_analysis.save(save_dir + f"/{window_size}/{env}_scale")
                seeds_analysis.save(save_dir + f"/{window_size}/{env}_seed")


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    root_dir = args.root_dir if args.root_dir else DEFAULT_DATA_FOLDER
    save_dir = args.save_dir if args.save_dir else DEFAULT_SAVE_DIR
    if os.path.exists(root_dir):
        # bsuite_graphing(root_dir, save_dir)
        graph_custom_env(root_dir, save_dir)
    else:
        raise OSError(f"Folder {root_dir} does not exist.")
