import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.utils import get_save_path

sns.set_style("dark")

#
# def viz_attention(weights, episode, step, total_steps, avg_attend, total):
#     """
#     Weights shape: (batch_size, target_seq, source_seq)
#     """
#     w = np.array(weights[0, -1, :])
#     x = np.arange(w.shape[0])
#     window_len = len(x)
#     colors = ["cyan"] * window_len
#     top_attend = np.argmax(w)
#     context_position = window_len - step - 1
#     if top_attend == context_position:
#         colors[top_attend] = "green"
#     else:
#         colors[top_attend] = "orange"
#         colors[context_position] = "red"
#     fig, ax1 = plt.subplots()
#
#     left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
#     ax2 = fig.add_axes([left, bottom, width, height])
#     ax2.set_xlim(0, total)
#     ax2.set_ylim(0, 1)
#     ax2.set_xlabel("Steps")
#     ax2.set_ylabel("Frequency correct")
#     ax2.plot(np.arange(len(avg_attend)), avg_attend)
#
#     ax1.bar(x, w, color=colors)
#     ax1.set_ylim(0, 2)
#     ax1.set_ylabel("Attention weights")
#     ax1.set_xlabel("States in sequence")
#     ax1.set_title(f"Episode {episode}, step {step}, total steps {total_steps}")
#     plt.savefig("plots/fig{:03d}.png".format(total_steps))
#     plt.close()
#     return top_attend == context_position


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
        w = w.detach().numpy()
        x = np.arange(w.shape[0])

        fig, ax1 = plt.subplots()

        ax1.bar(x, w, color="blue")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Attention weights")
        ax1.set_xlabel("States in sequence")
        ax1.set_title(f"Episode {eps_num}")
        plt.savefig(plot_save_dir + env_id + "_Eps_{:03d}.png".format(eps_num + 1))
        plt.close()
