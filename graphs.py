import os
import hydra
from omegaconf import DictConfig, ListConfig
from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from environment.custom_memory import score


def plot_training_curve():
    pass


def plot_evaluation_curve():
    pass


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
    window: int,
    dir: str,
):
    env_nums = list(map(str, range(13)))
    experiments = {}
    learning_threshold = 0.75
    for memory in memory_models:
        experiments[memory] = f"{window}/{memory}/data/{dir}/"

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
    save_dir = f"{window}/plots/bsuite/"
    plt.savefig(f"{save_dir}/custom_memory_scale.png")
    plt.close()

    # plot_custom_env_score(save_dir, experiments.keys(), regret_list, learning_threshold)


def plot_bsuite_graph(
    memory_models: ListConfig,
    envs: ListConfig,
    window: int,
    dir: str,
):
    if "memory_custom" in envs:
        plot_custom_env(memory_models, window, dir)
        envs.remove("memory_custom")

    experiment = {}
    for memory in memory_models:
        experiment[memory] = f"{window}/{memory}/data/{dir}/"

    DF, SWEEP_VARS = csv_load.load_bsuite(experiment)
    BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)

    for env in envs:
        # Plots specialized to the experiment
        result = summary_analysis.plot_single_experiment(BSUITE_SCORE, env, SWEEP_VARS)

        save_dir = f"{window}/plots/bsuite/"

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

            learning_analysis.save(save_dir + f"/{window}/{env}_learning")
            scale_analysis.save(save_dir + f"/{window}/{env}_scale")
            seeds_analysis.save(save_dir + f"/{window}/{env}_seed")


@hydra.main(config_path="configs/", config_name="graphs")
def main(cfg: DictConfig):
    print("Plotting results...")
    if cfg.plot_bsuite:
        plot_bsuite_graph(cfg.memory_models, cfg.envs, cfg.window, cfg.dir)


if __name__ == "__main__":
    main()
