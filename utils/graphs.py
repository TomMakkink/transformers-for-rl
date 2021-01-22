import os
from omegaconf import ListConfig
from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import get_sweep_from_bsuite_id
from environment.custom_memory import score
import torch
import seaborn as sns


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


def plot_result_curve(
    x: list,
    y: list,
    y_labels: list,
    x_axis_label: str,
    y_axis_label: str,
    title: str,
    save_dir: str,
):
    fig, ax = plt.subplots()

    for x_data, y_data, label in zip(x, y, y_labels):
        ax.plot(x_data, y_data, label=label)

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label, title=title)
    ax.grid()
    ax.grid(b=True, which="major")
    ax.legend()

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}{title.replace(' ', '_')}.png"
    fig.savefig(save_path)
    plt.close()


def plot_result_bar(x, y, xlabel: str, ylabel: str, title: str, save_dir: str):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    ax.grid(b=True, which="major")
    ax.set_ylim([0, 1])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}{title.replace(' ', '_')}.png"
    fig.savefig(save_path)
    plt.close()


def plot_training_results(memory_models: ListConfig, envs: ListConfig, seed: int):
    experiments = get_log_data(
        memory_models, envs, seed, file_suffix="_log", dir="training"
    )

    for env_id, training_data in experiments.items():
        x = []
        ave_score = []
        loss = []
        y_labels = []
        for memory_model, results in training_data.items():
            x.append(results["Episode"])
            ave_score.append(results["Average_Score"])
            loss.append(results["Loss"])
            y_labels.append(memory_model)

        save_dir = f"{seed}/plots/training/"
        plot_result_curve(
            x,
            y=ave_score,
            y_labels=y_labels,
            x_axis_label="Episodes",
            y_axis_label="Average Score",
            title=f"{env_id} Training",
            save_dir=save_dir,
        )


def plot_evaluation_results(memory_models: ListConfig, envs: ListConfig, seed: int):
    experiments = get_log_data(
        memory_models, envs, seed, file_suffix="_log", dir="eval"
    )

    for env_id, eval_data in experiments.items():
        x = []
        y = []
        y_labels = []
        for memory_model, results in eval_data.items():
            x.append(memory_model)
            y.append(results["Average_Score"])
            y_labels.append(memory_model)

        save_dir = f"{seed}/plots/eval/"

        plot_result_bar(
            x,
            y,
            "Memory Models",
            "Average Score",
            title=f"{env_id} Evaluation",
            save_dir=save_dir,
        )


def plot_custom_env_score(save, names, regret_list, learning_threshold):
    fig, ax = plt.subplots()
    scores = []
    for i, regret in enumerate(regret_list):
        score = np.mean(np.array(regret) < learning_threshold)
        scores.append(score)

    ax.bar(names, scores)
    ax.title()
    ax.set_xlabels()
    plt.savefig(f"{save}/custom_memory_results.png")
    plt.close()


def plot_custom_env(
    memory_models: ListConfig,
    seed: int,
    dir: str,
):
    env_nums = list(map(str, range(13)))
    experiments = {}
    learning_threshold = 0.75
    for memory in memory_models:
        experiments[memory] = f"{seed}/{memory}/data/{dir}/"

    fig, axs = plt.subplots(3, 3, sharey=True, figsize=(16, 16))
    x = np.arange(len(env_nums))
    row_index = 0
    regret_list = []
    for i, (name, file_dir) in enumerate(experiments.items()):
        regret = []
        for env_num in env_nums:
            file = f"{file_dir}bsuite_id_-_memory_custom-{env_num}.csv"
            env_df = pd.read_csv(file)
            env_regret = score(env_df)
            regret.append(env_regret)

        colors = ["blue" if x < learning_threshold else "red" for x in regret]
        axs[row_index // 3, i % 3].scatter(x, regret, s=200, c=colors)
        axs[row_index // 3, i % 3].set_title(f"{name}")
        # axs[row_index // 3, i % 3].set_xlabel("Environments")
        # axs[row_index // 3, i % 3].set_ylabel("Average Regret at 10000 episodes")
        # axs[row_index // 3, i % 3].set_xticks(x)

        regret_list.append(regret)
        row_index += 1

    fig.delaxes(axs[2, 1])
    fig.delaxes(axs[2, 2])

    # Set common labels
    fig.text(0.5, 0.04, "Environments", ha="center", va="center")
    fig.text(
        0.06,
        0.5,
        "Average Regret at 10000 episodes",
        ha="center",
        va="center",
        rotation="vertical",
    )

    fig.suptitle("Custom Memory Environment", fontsize=16)
    # y = np.repeat(LEARNING_THRESH, len(ENV_NUMS))
    # ax.plot(x, y, '--')
    # axs.legend()
    save_dir = f"{seed}/plots/bsuite"
    plt.savefig(f"{save_dir}/custom_memory_scale.png")
    plt.close()

    # plot_custom_env_score(save_dir, experiments.keys(), regret_list, learning_threshold)


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
    memory_models: ListConfig, envs: ListConfig, seed: int, plot_frequency: int
):
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
        ax.set(ylim=(0, 1))
        ax.set_title(eps)
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Attention Weighting")
        fig = ax.get_figure()
        fig.savefig(f"{save_dir}eps_{eps}_attn_weights.png")
        plt.close()
