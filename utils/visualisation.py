import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.utils import get_save_path
import pickle
import pandas as pd

sns.set_style("dark")

PLOT_EVERY = 1000


def viz_forget_activation(forget_activation, env_id, agent_name, window_size,
                          memory='lstm'):
    plot_save_dir = get_save_path(window_size, agent_name, memory) + "plots/"
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)

    episode_len = len(forget_activation[0])
    for eps_num, episode in enumerate(forget_activation):
        fig, ax1 = plt.subplots()

        f_t_mean = [data[2] for data in episode]
        f_t_std = [data[3] for data in episode]
        ax1.bar(range(episode_len), f_t_mean, yerr=f_t_std)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Mean Forget Gate Activation")
        ax1.set_xlabel("States in sequence")
        ax1.set_title(f"Episode {eps_num + 1}")
        # TODO: Check if folder exists
        plt.savefig(plot_save_dir + env_id + "_Eps_{:03d}.png".format(eps_num + 1))
        plt.close()


def plot_lstm_forget_activation_heat_map(viz_data, env_id, agent_name,
                                         window_size, memory='lstm'):
    plot_save_dir = get_save_path(window_size, agent_name, memory) + "plots/"
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)

    for eps_num, episode in enumerate(viz_data):
        if (eps_num + 1) % PLOT_EVERY != 0:
            continue
        print(f"Plotting episode:{eps_num + 1}")
        l = np.asarray([np.mean(t["full_f_t_activations"], axis=1) for t in episode])
        l = np.transpose(l)

        fig, ax = plt.subplots(figsize=(10, 10))  # Sample figsize in inches

        sns.heatmap(l, vmin=0, vmax=1, linewidths=.5, ax=ax)
        ax.set_title(f"Episode {eps_num + 1}")
        figure = ax.get_figure()
        figure.savefig(
            plot_save_dir + env_id + "_Eps_{:06d}_heatmap.png".format(eps_num + 1))
        plt.close()


def plot_lstm_gates(gate_activations, env_id, agent_name, window_size, memory='lstm'):
    plot_save_dir = get_save_path(window_size, agent_name, memory) + "plots/"
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)

    episode_len = len(gate_activations[0])
    for eps_num, episode in enumerate(gate_activations):
        if (eps_num + 1) % PLOT_EVERY != 0:
            continue
        print(f"Plotting episode:{eps_num + 1}")
        fig, axs = plt.subplots(3, 1, sharey=True)

        f_t_mean = [data["forget_gate"][2] for data in episode]
        f_t_std = [data["forget_gate"][3] for data in episode]

        i_t_mean = [data["input_gate"][2] for data in episode]
        i_t_std = [data["input_gate"][2] for data in episode]

        o_t_mean = [data["output_gate"][2] for data in episode]
        o_t_std = [data["output_gate"][3] for data in episode]

        x = range(episode_len)

        axs[0].bar(x, f_t_mean, yerr=f_t_std)
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel("Forget Gate")
        axs[0].set_xlabel("First state in sequence")

        axs[1].bar(x, i_t_mean, yerr=i_t_std)
        axs[1].set_ylabel("Input Gate")
        axs[1].set_xlabel("First state in sequence")

        axs[2].bar(x, o_t_mean, yerr=o_t_std)
        axs[2].set_ylabel("Output Gate")
        axs[2].set_xlabel("First state in sequence")

        axs[0].set_title(f"Episode {eps_num + 1}")
        plt.savefig(plot_save_dir + env_id + "_Eps_{:06d}.png".format(eps_num + 1))
        plt.close()


def viz_attention(weights, env_id, agent_name, window_size, memory):
    """
    Weights shape: List[(layer, batch_size, target_seq, source_seq)]
    """
    plot_save_dir = get_save_path(window_size, agent_name, memory) + "plots/"
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)

    for eps_num, weight in enumerate(weights):
        if (eps_num + 1) % PLOT_EVERY != 0:
            continue
        print(f"Plotting episode:{eps_num + 1}")

        last_timestep = weight[-1]  # For not plot last timestep only
        w = last_timestep[0, 0, -1, :]  # Last layer only
        w = w.detach().cpu().numpy()
        x = np.arange(w.shape[0])

        fig, ax1 = plt.subplots()

        ax1.bar(x, w, color="blue")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Attention weights")
        ax1.set_xlabel("States in sequence")
        ax1.set_title(f"Episode {eps_num}")
        plt.savefig(plot_save_dir + env_id + "_Eps_{:06d}.png".format(eps_num + 1))
        plt.close()


def plot_rewards(env_id, save_dir, rolling_window=100):
    reward_data_file = save_dir + f"{env_id}_rewards.pt"
    fileObject = open(reward_data_file, 'rb')
    rewards = pickle.load(fileObject)
    fileObject.close()

    plot_save_dir = save_dir + "plots/"
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)

    df = pd.DataFrame(data=rewards, columns=["Rewards"])
    df['Rewards'] = df['Rewards'].rolling(rolling_window).mean()

    fig, ax = plt.subplots()
    sns.lineplot(x=df.index, y="Rewards", data=df, ax=ax)
    ax.set_title(f"Total Rewards for Episodes (Rolling mean window = {rolling_window})")
    figure = ax.get_figure()
    figure.savefig(f"{plot_save_dir}{env_id}_rewards.png")
    plt.close()
