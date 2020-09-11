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

sns.set()

DEFAULT_SAVE_DIR = "graphs"
DEFAULT_DATA_FOLDER = "results"
ALGORITHMS = ["A2C"]
MODELS = ["mha", "rmha", "gmha", "lstm", "none"]
ENVS = ["memory_size"]  # "memory_len",
ENVS_NUM = list(map(str, range(17)))
SEEDS = ["28", "61", "39"]  # , "78", "72", "46", "61", "71", "93", "44"]
COLOURS = ["blue", "green", "red", "purple", "black", "orange"]
WINDOW = ["10"]

# Results Directory Structure:
# - results
#     - env
#         - env_num (e.g. 0, as in mem_len/0)
#             - algo
#                 - model
#                     - seed


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
                for model in MODELS:
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


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis


def bsuite_graphing(root_dir, save_dir):
    SAVE_PATH_NONE = root_dir + f"/a2c/None/"
    SAVE_PATH_GMHA = root_dir + f"/a2c/gmha/"
    experiments = {"none": SAVE_PATH_NONE}
    DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
    BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
    print(BSUITE_SCORE)

    # # @title As well as plots specialized to the experiment
    # umbrella_length_df = DF[DF.bsuite_env == "umbrella_length"].copy()
    result = summary_analysis.plot_single_experiment(BSUITE_SCORE, "memory_size")
    result.save(save_dir + "/Summary")

    # learn = umbrella_length_analysis.plot_learning(umbrella_length_df)
    # learn.save(save_dir + "/umbrella_length_learning")


#     experiments

# experiments = {'dqn': SAVE_PATH_DQN, 'rand': SAVE_PATH_RAND}
# DF, SWEEP_VARS = csv_load.load_bsuite(experiments)


if __name__ == "__main__":
    root_dir = args.root_dir if args.root_dir else DEFAULT_DATA_FOLDER
    save_dir = args.save_dir if args.save_dir else DEFAULT_SAVE_DIR
    if os.path.exists(root_dir):
        # process_data(root_dir, save_dir)
        bsuite_graphing(root_dir, save_dir)
    else:
        raise OSError(f"Folder {root} does not exist.")
