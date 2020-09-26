import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.utils import get_save_path

sns.set_style("dark")


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


def plot_lstm_gates(gate_activations, env_id, agent_name, window_size, memory='lstm'):
    plot_save_dir = get_save_path(window_size, agent_name, memory) + "plots/"
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)

    episode_len = len(gate_activations[0])
    for eps_num, episode in enumerate(gate_activations):
        if eps_num % 10 != 0:
            continue

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
        plt.savefig(plot_save_dir + env_id + "_Eps_{:03d}.png".format(eps_num + 1))
        plt.close()


def viz_attention(weights, env_id, agent_name, window_size, memory):
    """
    Weights shape: List[(layer, batch_size, target_seq, source_seq)]
    """
    plot_save_dir = get_save_path(window_size, agent_name, memory) + "plots/"
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)

    for eps_num, weight in enumerate(weights):
        if eps_num % 10 != 0:
            continue

        last_timestep = weight[-1]  # For not plot last timestep only
        w = last_timestep[0, 0, -1, :] # Last layer only
        w = w.detach().cpu().numpy()
        x = np.arange(w.shape[0])

        fig, ax1 = plt.subplots()

        ax1.bar(x, w, color="blue")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Attention weights")
        ax1.set_xlabel("States in sequence")
        ax1.set_title(f"Episode {eps_num}")
        plt.savefig(plot_save_dir + env_id + "_Eps_{:03d}.png".format(eps_num + 1))
        plt.close()
