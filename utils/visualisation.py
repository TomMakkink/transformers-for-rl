import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("dark")


def viz_attention(weights, episode, step, total_steps, avg_attend, total):
    """
    Weights shape: (batch_size, target_seq, source_seq)
    """
    w = np.array(weights[0, -1, :])
    x = np.arange(w.shape[0])
    window_len = len(x)
    colors = ["cyan"] * window_len
    top_attend = np.argmax(w)
    context_position = window_len - step - 1
    if top_attend == context_position:
        colors[top_attend] = "green"
    else:
        colors[top_attend] = "orange"
        colors[context_position] = "red"
    fig, ax1 = plt.subplots()

    left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.set_xlim(0, total)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Frequency correct")
    ax2.plot(np.arange(len(avg_attend)), avg_attend)

    ax1.bar(x, w, color=colors)
    ax1.set_ylim(0, 2)
    ax1.set_ylabel("Attention weights")
    ax1.set_xlabel("States in sequence")
    ax1.set_title(f"Episode {episode}, step {step}, total steps {total_steps}")
    plt.savefig("plots/fig{:03d}.png".format(total_steps))
    plt.close()
    return top_attend == context_position


def viz_attention_mem(weights, episode, step):
    """
    Weights shape: (batch_size, target_seq, source_seq)
    """
    w = np.array(weights[0, -1, :])
    w = np.pad(w, (6 - len(w), 0), "constant")
    x = np.arange(w.shape[0])

    fig, ax1 = plt.subplots()

    ax1.bar(x, w, color="blue")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Attention weights")
    ax1.set_xlabel("States in sequence")
    ax1.set_title(f"Episode {episode}, step {step}")
    plt.savefig("plots/fig_{:03d}_{:03d}.png".format(episode, step))
    plt.close()
