from itertools import zip_longest
import numpy as np
import os
from os import listdir, walk
from os.path import isfile, join, splitext
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
    "None",
    "rezero",
    "vanilla",
    "xl",
]

ENVS = ["umbrella_distract", "umbrella_length"]
ENVS_NUM = list(map(str, range(17)))
SEEDS = ["28", "61", "39"]  # , "78", "72", "46", "61", "71", "93", "44"]
COLOURS = ["blue", "green", "red", "purple", "black", "orange"]
WINDOW_SIZES = ["4"]

#   - results
#       - window
#           - agent
#               - memory


def process_data(root, save):
    colour_index = 0

    for env in ENVS:
        for env_num in ENVS_NUM:
            fig, ax = plt.subplots()
            ax.set_title(f"{env}/{env_num}")
            ax.set_xlabel("Episodes")
            ax.set_ylabel("Episode Return")
            for algo in ALGORITHMS:
                colour_index = 0
                for model in MEMORY:
                    returns = []
                    episodes = []
                    for seed in SEEDS:
                        results_dir = "/".join(
                            [root, env, env_num, algo, model, seed, WINDOW[0]]
                        )
                        if not os.path.isdir(results_dir):
                            os.makedirs(results_dir)
                        # if os.path.exists(results_dir):
                        results_csv = listdir(results_dir)
                        if len(results_csv) == 1:
                            results_path = "/".join([results_dir, results_csv[0]])
                            data = np.genfromtxt(
                                results_path, dtype=float, delimiter=",", names=True
                            )
                            # print(
                            #     f"Shape of episode_return: {data['episode_return'].shape}")
                            returns.append(data["episode_return"])
                            episodes.append(data["episode"])

                    # Assume episodes are all the same
                    x = np.array(episodes[0])
                    y = calculate_mean(returns)
                    std = calculate_std(returns)
                    if len(x) == len(y):
                        ax.plot(
                            x, y, "-", color=COLOURS[colour_index], label=f"{model}"
                        )
                        # ax.fill_between(
                        #     x, y - std, y + std, color=COLOURS[colour_index], alpha=0.2
                        # )
                        colour_index = colour_index + 1
                        ax.set_xscale("log")
                    else:
                        print(
                            f"Skippping file due to mismatch sizes between x: {len(x)} and y: {len(y)}"
                        )
                ax.legend()
                if not os.path.isdir(save):
                    os.makedirs(save)
                fig.savefig(f"{save}/{algo}_{env}_{env_num}.pdf", format="pdf")


def calculate_mean(data):
    return np.nanmean(np.array(list(zip_longest(*data)), dtype=float), axis=1)


def calculate_std(data):
    return np.nanstd(np.array(list(zip_longest(*data)), dtype=float), axis=1)


import pandas as pd
from bsuite import sweep
import glob
import os
from bsuite.logging import csv_logging
from bsuite.logging import logging_utils
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()


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


if __name__ == "__main__":
    root_dir = args.root_dir if args.root_dir else DEFAULT_DATA_FOLDER
    save_dir = args.save_dir if args.save_dir else DEFAULT_SAVE_DIR
    if os.path.exists(root_dir):
        # process_data(root_dir, save_dir)
        bsuite_graphing(root_dir, save_dir)
    else:
        raise OSError(f"Folder {root} does not exist.")
