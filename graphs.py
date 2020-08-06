from itertools import zip_longest
import numpy as np
import os
from os import listdir, walk
from os.path import isfile, join, splitext
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


DEFAULT_SAVE_DIR = 'graphs'
DEFAULT_DATA_FOLDER = 'results'
ALGORITHMS = ['A2C']
MODELS = ['rezero', 'gtrxl', 'vanilla', 'xl', 'LSTM', 'none']
ENVS = ['memory_len', 'memory_size', 'cartpole']
ENVS_NUM = ['0', '1', '2', '3', '4', '5']
SEEDS = ['28', '61', '39', '78', '72', '46', '61', '71', '93', '44']
COLOURS = ['blue', 'green', 'red', 'purple', 'black', 'orange']

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
                        results_dir = '/'.join([root, env,
                                                env_num, algo, model, seed])
                        if os.path.exists(results_dir):
                            results_csv = listdir(results_dir)
                            if len(results_csv) == 1:
                                results_path = '/'.join([results_dir,
                                                         results_csv[0]])
                                data = np.genfromtxt(
                                    results_path, dtype=float, delimiter=',', names=True)
                                # print(
                                #     f"Shape of episode_return: {data['episode_return'].shape}")
                                returns.append(data['episode_return'])
                                episodes.append(data["episode"])

                    # Assume episodes are all the same
                    print(len(episodes[0]))
                    x = np.array(episodes[0])
                    y = calculate_mean(returns)
                    std = calculate_std(returns)
                    if len(x) == len(y):
                        ax.plot(
                            x, y, '-', color=COLOURS[colour_index], label=f"{model}")
                        # ax.fill_between(x, y - std, y + std,
                        #                 color=COLOURS[colour_index], alpha=0.2)
                        colour_index = colour_index + 1
                    else:
                        print(
                            f"Skippping file due to mismatch sizes between x: {len(x)} and y: {len(y)}")
                ax.legend()
                fig.savefig(
                    f"{save}/{algo}_{env}_{env_num}.pdf", format='pdf')


def calculate_mean(data):
    return np.nanmean(np.array(list(zip_longest(*data)), dtype=float), axis=1)


def calculate_std(data):
    return np.nanstd(np.array(list(zip_longest(*data)), dtype=float), axis=1)


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    root_dir = args.root_dir if args.root_dir else DEFAULT_DATA_FOLDER
    save_dir = args.save_dir if args.save_dir else DEFAULT_SAVE_DIR
    if os.path.exists(root_dir):
        process_data(root_dir, save_dir)
    else:
        raise OSError(f"Folder {root} does not exist.")
