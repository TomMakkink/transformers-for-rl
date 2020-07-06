import numpy as np
import scipy.signal
import torch


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p is not None:
            print(f"Layer name: {n}")
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    print(f"Average grads: {ave_grads}")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def set_random_seed(seed: int, use_cuda: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    return device

def process_obs(obs, device):
    obs = obs.squeeze()
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def log_to_comet_ml(experiment, total_time_steps, mean_episode_returns, mean_episode_length, loss_actor, loss_critic,
                    loss, entropy, kl):
    experiment.log_metric('Mean Episode Reward', mean_episode_returns, step=total_time_steps)
    experiment.log_metric('Mean Episode Length', mean_episode_length, step=total_time_steps)
    experiment.log_metric('Actor Loss', loss_actor.item(), step=total_time_steps)
    experiment.log_metric('Critic Loss', loss_critic.item(), step=total_time_steps)
    experiment.log_metric('Loss', loss.item(), step=total_time_steps)
    experiment.log_metric('Entropy', entropy, step=total_time_steps)
    experiment.log_metric('Kl', kl, step=total_time_steps)


def log_to_tensorboard(writer, total_time_steps, mean_episode_returns, mean_episode_length, loss_actor, loss_critic, loss, ent, kl):
    writer.add_scalar('Mean Episode Reward',
                            mean_episode_returns, total_time_steps)
    writer.add_scalar('Mean Episode Length',
                            mean_episode_length, total_time_steps)
    writer.add_scalar('Loss/Loss', loss, total_time_steps)
    writer.add_scalar(
        'Loss/Actor Loss', loss_actor, total_time_steps)
    writer.add_scalar(
        'Loss/Critic Loss', loss_critic, total_time_steps)
    writer.add_scalar('Extra/Kl', kl, total_time_steps)
    writer.add_scalar('Extra/Entropy', ent, total_time_steps)


def log_to_screen(total_episodes, total_time_steps, mean_episode_returns, mean_episode_length, loss_actor, loss_critic, loss, ent, kl):
    print("------------------------------")
    print(f"Mean episode returns: {mean_episode_returns:.2f}")
    print(f"Mean episode length: {mean_episode_length}")
    print(f"Loss actor: {loss_actor:.2f}")
    print(f"Loss critic: {loss_critic:.2f}")
    print(f"Loss: {loss:.2f}")
    print(f"Ent: {ent:.2f}")
    print(f"KL: {kl:.2f}")
    print(f"Timestep: {total_time_steps}")
    print(f"Episodes: {total_episodes}")
    print("------------------------------")
