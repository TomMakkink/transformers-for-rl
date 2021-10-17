import os
from typing import List, Optional

from omegaconf import ListConfig
from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from utils import get_sweep_from_bsuite_id
from environment.custom_memory import score
import torch
import seaborn as sns
import warnings

sns.set_style("whitegrid")


def get_log_data(
    memory_models: ListConfig,
    envs: ListConfig,
    seed: int,
    file_suffix: str = "_log",
    file_type: str = "csv",
    dir: str = "training",
):
    assert file_type in ["csv", "pt"], f"File type {file_type} invalid."
    experiments = {}

    for env in envs:
        env_id_list = get_sweep_from_bsuite_id(env)
        for env_id in env_id_list:
            env_id = env_id.replace("/", "-")
            experiments[env_id] = {}
            for memory in memory_models:
                path_to_file = (
                    f"{seed}/{memory}/data/{dir}/{env_id}{file_suffix}.{file_type}"
                )
                if file_type == "csv":
                    data = np.genfromtxt(
                        path_to_file, dtype=float, delimiter=",", names=True
                    )
                else:
                    data = torch.load(path_to_file)
                experiments[env_id][memory] = data

    return experiments


def get_experiments_df(
    memory_models: ListConfig,
    envs: ListConfig,
    seeds: ListConfig,
    dir: str,
    column_names: Optional[List] = None,
):
    df_data = []
    for seed in seeds:
        for memory in memory_models:
            for env in envs:
                env_id_list = get_sweep_from_bsuite_id(env)
                for env_id in env_id_list:
                    env_id = env_id.replace("/", "-")
                    path_to_file = f"{seed}/{memory}/data/{dir}/{env_id}_log.csv"
                    if not os.path.exists(path_to_file):
                        warnings.warn(f"Path {path_to_file} doesn't exist. Skipping.")
                        continue
                    data = pd.read_csv(
                        path_to_file,
                        names=column_names,
                        index_col=None,
                        header=0,
                    )
                    data["Memory"] = memory
                    data["Experiment"] = int(env_id.split("-")[-1])
                    data["Environment"] = env_id.split("-")[0]
                    df_data.append(data)
    df = pd.concat(df_data, axis=0, ignore_index=True)
    return df


def plot_training_results(
    memory_models: ListConfig, envs: ListConfig, seeds: List[int]
):
    # Set figure aesthetics back to default
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_style("whitegrid")

    experiments = get_experiments_df(
        memory_models,
        envs,
        seeds,
        dir="training",
        column_names=["Episode", "Mean Return", "Loss"],
    )
    save_dir = f"plots/training/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for env_type, env_data in experiments.groupby("Environment"):
        env_data["Context Length"] = get_context_length(env_type, env_data)
        env_data["Context Size"] = get_context_size(env_type, env_data)

        if env_type in "memory_len":
            col = "Context Length"
        elif env_type in "memory_size":
            col = "Context Size"
        else:
            col = "Experiment"

        env_data.Memory = env_data.Memory.apply(lambda x: x.replace(" ", "\n"))

        grid = sns.FacetGrid(env_data, col=col, hue="Memory", col_wrap=2)
        grid.map_dataframe(sns.lineplot, x="Episode", y="Mean Return", ci="sd")
        grid.add_legend(title="")
        grid.set_axis_labels("Episode", "Mean Return")
        grid.tight_layout()
        grid.savefig(f"{save_dir}{env_type}_train_summary.png", dpi=300)
        plt.close()


def get_env_title(env: str) -> str:
    return {
        "memory_len": "Memory Length",
        "memory_size": "Memory Size",
        "memory_custom": "Distributed Memory",
    }.get(env)


def plot_evaluation_results(
    memory_models: ListConfig, envs: ListConfig, seeds: ListConfig
):
    # Set figure aesthetics back to default
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_style("whitegrid")

    experiments = get_experiments_df(
        memory_models,
        envs,
        seeds,
        dir="eval",
        column_names=["Episode", "Mean Return"],
    )

    plots_save_dir = f"plots/eval/"
    tables_save_dir = f"tables/eval/"

    for save_dir in [plots_save_dir, tables_save_dir]:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    for env_type, env_data in experiments.groupby("Environment"):
        env_data["Context Length"] = get_context_length(env_type, env_data)
        env_data["Context Size"] = get_context_size(env_type, env_data)
        env_data["Context"] = list(
            zip(env_data["Context Length"], env_data["Context Size"])
        )

        save_results_to_latex_table(env_type, env_data, tables_save_dir)
        plot_eval_summary(env_type, env_data, plots_save_dir)
        plot_eval_relative_to_lstm(env_type, env_data, plots_save_dir)


def _memory_len_context_length(experiment_num: int) -> int:
    assert (
        experiment_num >= 0 and experiment_num <= 22
    ), f"Experiment index {experiment_num} out of range."

    # How they calculate context length in deepmind bsuite
    # https://github.com/deepmind/bsuite/blob/master/bsuite/experiments/memory_len/sweep.py
    _log_spaced = []
    _log_spaced.extend(range(1, 11))
    _log_spaced.extend([12, 14, 17, 20, 25])
    _log_spaced.extend(range(30, 105, 10))

    return _log_spaced[experiment_num]


def _custom_memory_context_length(experiment_num: int) -> int:
    assert (
        experiment_num >= 0 and experiment_num <= 12
    ), f"Experiment index {experiment_num} out of range."

    if experiment_num == 0:
        return 3
    elif experiment_num < 3:
        return 5
    elif experiment_num < 7:
        return 10
    elif experiment_num <= 12:
        return 30


def get_context_length(env_type: str, env_data: pd.DataFrame) -> pd.Series:
    if "memory_len" in env_type:
        return env_data.Experiment.map(_memory_len_context_length)
    elif "memory_size" in env_type:
        return 1
    else:
        return env_data.Experiment.map(_custom_memory_context_length)


def _memory_size_context_size(experiment_num: int) -> int:
    assert (
        experiment_num >= 0 and experiment_num <= 16
    ), f"Experiment index {experiment_num} out of range."

    _log_spaced = []
    _log_spaced.extend(range(1, 11))
    _log_spaced.extend([12, 14, 17, 20, 25])
    _log_spaced.extend(range(30, 50, 10))

    return _log_spaced[experiment_num]


def _custom_memory_context_size(experiment_num: int) -> int:
    assert (
        experiment_num >= 0 and experiment_num <= 12
    ), f"Experiment index {experiment_num} out of range."

    if experiment_num in [0, 1, 3, 7]:
        return 3
    elif experiment_num in [2, 4, 8]:
        return 5
    elif experiment_num in [5, 9]:
        return 7
    elif experiment_num in [6, 10]:
        return 9
    elif experiment_num == 11:
        return 17
    elif experiment_num == 12:
        return 25


def get_context_size(env_type: str, env_data: pd.DataFrame) -> pd.Series:
    if "memory_size" in env_type:
        return env_data.Experiment.map(_memory_size_context_size)
    elif "memory_len" in env_type:
        return 1
    else:
        return env_data.Experiment.map(_custom_memory_context_size)


def save_results_to_latex_table(env_type: str, env_data: pd.DataFrame, save_dir: str):
    mean_pivot_table = pd.pivot_table(
        env_data,
        values="Mean Return",
        index=["Context Length", "Context Size"],
        columns=["Memory"],
        aggfunc=np.mean,
    )

    std_pivot_table = pd.pivot_table(
        env_data,
        values="Mean Return",
        index=["Context Length", "Context Size"],
        columns=["Memory"],
        aggfunc=np.std,
    )

    std_pivot_table.columns = std_pivot_table.columns.map(lambda x: f"{x}_std")

    combined_df = pd.concat([mean_pivot_table, std_pivot_table], axis=1)
    combined_df.loc["Mean"] = combined_df.mean()

    combined_df.sort_index(axis=1, inplace=True)
    combined_df.round(3).to_latex(f"{save_dir}mean_score_{env_type}.tex")


def plot_eval_summary(env_type: str, env_data: pd.DataFrame, save_dir: str):
    if "memory_len" in env_type:
        x = "Context Length"
    elif "memory_size" in env_type:
        x = "Context Size"
    else:
        x = "Context"

    ax = sns.barplot(
        data=env_data,
        x=x,
        y="Mean Return",
        hue="Memory",
        ci="sd",
        errwidth=0.75,
    )
    ax.legend(loc="upper center", fontsize="x-small", ncol=2)
    if env_type == "memory_custom":
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(f"{save_dir}{env_type}_eval_summary.png", dpi=300)
    plt.close()


def plot_eval_relative_to_lstm(env_type: str, env_data: pd.DataFrame, save_dir: str):
    assert (
        "LSTM" in env_data.Memory.values
    ), "LSTM scores missing. Cannot calculate relative score."

    if "memory_len" in env_type:
        index = "Context Length"
    elif "memory_size" in env_type:
        index = "Context Size"
    else:
        index = "Context"

    relative_score_df = pd.pivot_table(
        env_data, values="Mean Return", index=[index], columns=["Memory"]
    )
    relative_score_df = relative_score_df.subtract(relative_score_df.LSTM, axis="index")

    if "No Memory" in env_data.columns:
        relative_score_df.drop(["LSTM", "No Memory"], axis="columns", inplace=True)
    else:
        relative_score_df.drop(["LSTM"], axis="columns", inplace=True)
    relative_score_df.sort_index(inplace=True)

    ax = relative_score_df.plot.bar(
        rot=0,
        xlabel=index,
        ylabel="Mean Return relative to an LSTM baseline",
        legend=True,
    )
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(f"{save_dir}{env_type}_relative_eval_summary.png", dpi=300)
    plt.close()


def plot_custom_env_score(save, names, regret_list, learning_threshold):
    fig, ax = plt.subplots()
    scores = []
    for i, regret in enumerate(regret_list):
        score = np.mean(np.array(regret) < learning_threshold)
        scores.append(score)

    ax.bar(names, scores)
    ax.title()
    ax.set_xlabels()
    plt.savefig(f"{save}/custom_memory_results.png", dpi=300)
    plt.close()


# def plot_custom_env(
#     memory_models: ListConfig,
#     seed: int,
#     dir: str,
# ):
#     env_nums = list(map(str, range(13)))
#     experiments = {}
#     learning_threshold = 0.75
#     for memory in memory_models:
#         experiments[memory] = f"{seed}/{memory}/data/{dir}/"
#
#     fig, axs = plt.subplots(3, 3, sharey=True, figsize=(16, 16))
#     x = np.arange(len(env_nums))
#     row_index = 0
#     regret_list = []
#     for i, (name, file_dir) in enumerate(experiments.items()):
#         regret = []
#         for env_num in env_nums:
#             file = f"{file_dir}bsuite_id_-_memory_custom-{env_num}.csv"
#             env_df = pd.read_csv(file)
#             env_regret = score(env_df)
#             regret.append(env_regret)
#
#         colors = ["blue" if x < learning_threshold else "red" for x in regret]
#         axs[row_index // 3, i % 3].scatter(x, regret, s=200, c=colors)
#         axs[row_index // 3, i % 3].set_title(f"{name}")
#         # axs[row_index // 3, i % 3].set_xlabel("Environments")
#         # axs[row_index // 3, i % 3].set_ylabel("Average Regret at 10000 episodes")
#         # axs[row_index // 3, i % 3].set_xticks(x)
#
#         regret_list.append(regret)
#         row_index += 1
#
#     fig.delaxes(axs[2, 1])
#     fig.delaxes(axs[2, 2])
#
#     # Set common labels
#     fig.text(0.5, 0.04, "Environments", ha="center", va="center")
#     fig.text(
#         0.06,
#         0.5,
#         "Average Regret at 10000 episodes",
#         ha="center",
#         va="center",
#         rotation="vertical",
#     )
#
#     fig.suptitle("Custom Memory Environment", fontsize=16)
#     # y = np.repeat(LEARNING_THRESH, len(ENV_NUMS))
#     # ax.plot(x, y, '--')
#     # axs.legend()
#     save_dir = f"{seed}/plots/bsuite"
#     plt.savefig(f"{save_dir}/custom_memory_scale.png")
#     plt.close()
#
#     # plot_custom_env_score(save_dir, experiments.keys(), regret_list, learning_threshold)


def plot_bsuite_graph(
    memory_models: ListConfig,
    envs: ListConfig,
    seed: int,
    dir: str,
):
    if "memory_custom" in envs:
        plot_custom_env(memory_models, seed, dir)
        envs.remove("memory_custom")

    experiment = {}
    for memory in memory_models:
        experiment[memory] = f"{seed}/{memory}/data/{dir}/"

    DF, SWEEP_VARS = csv_load.load_bsuite(experiment)
    BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)

    for env in envs:
        # Plots specialized to the experiment
        result = summary_analysis.plot_single_experiment(BSUITE_SCORE, env, SWEEP_VARS)

        save_dir = f"{seed}/plots/bsuite"

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        result.save(f"{save_dir}/{env}_results")

        if env in ["memory_len", "memory_size"]:
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

            learning_analysis.save(f"{save_dir}/{env}_learning")
            scale_analysis.save(f"{save_dir}/{env}_scale")
            seeds_analysis.save(f"{save_dir}/{env}_seed")


def plot_attention_weights(
    memory_models: ListConfig, envs: ListConfig, seeds: List[int], plot_frequency: int
):
    for seed in seeds:
        experiments = get_log_data(
            memory_models,
            envs,
            seed,
            file_suffix="_attn_weights",
            file_type="pt",
            dir="training",
        )

        data = []
        for env in experiments.keys():
            save_dir = f"{seed}/plots/attn_weights/{env}/"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            for memory in experiments[env].keys():
                for eps in range(len(experiments[env][memory])):
                    if eps % plot_frequency != 0:
                        continue
                    for t in range(len(experiments[env][memory][eps])):
                        w = experiments[env][memory][eps][t]
                        data.append([env, memory, eps, t, w])

        df = pd.DataFrame(
            data,
            columns=["env", "memory", "episode", "timestep", "attn_weight"],
        )

        for eps, eps_data in df.groupby("episode"):
            ax = sns.barplot(x="timestep", y="attn_weight", hue="memory", data=eps_data)
            ax.set_title(eps)
            ax.set_xlabel("Time steps")
            ax.set_ylabel("Attention Weighting")
            fig = ax.get_figure()
            fig.savefig(f"{save_dir}eps_{eps}_attn_weights.png")
            plt.close()
