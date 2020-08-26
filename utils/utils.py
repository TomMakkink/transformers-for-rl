import numpy as np
import scipy.signal
import torch
from gym.wrappers import TransformObservation

import bsuite
from agents.a2c import A2C
from agents.dqn import DQN

# from agents.ppo import PPO

from bsuite.utils import gym_wrapper
from configs.env_config import env_config
from configs.experiment_config import experiment_config
from configs.transformer_config import transformer_config
from models.actor_critic_lstm import ActorCriticLSTM
from models.actor_critic_mlp import ActorCriticMLP
from models.actor_critic_transformer import ActorCriticTransformer
import seaborn as sns
import matplotlib.pyplot as plt


def update_configs_from_args(args):
    if args.project:
        experiment_config.update({"project_name": args.project})
    if args.name:
        experiment_config.update({"experiment_name": args.name})
    if args.seed:
        experiment_config.update({"seed": args.seed})
    if args.memory in ["vanilla", "rezero", "linformer", "xl", "gtrxl"]:
        transformer_config.update({"transformer_type": args.memory})
    if args.env:
        env_config.update({"env": args.env})


def get_agent(agent_name: str):
    return {"a2c": A2C, "dqn": DQN}.get(agent_name, A2C)  # Defaults to A2C


# def algo_from_string(algo: str):
# if algo == "a2c":
#     algo = A2C
# elif algo == "ppo":
#     algo = PPO
# else:
#     print(f"Algorithm {args.algo} not implemented. Defaulting to PPO.")
#     algo = PPO
# return algo


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


def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_config["device"] = device


# def process_obs(obs, device):
#     obs = obs.squeeze()
#     return torch.as_tensor(obs, dtype=torch.float32, device=device)


def create_environment(
    agent, seed, memory, env=None,
):
    # build folder path to save data
    save_path = "results/"

    if env:
        save_path = save_path + env + "/"
    else:
        # TODO: Clean up
        # env = env_config["env"]
        save_path = save_path + env_config["env"] + "/"

    save_path = save_path + agent + "/" + memory + "/" + str(seed) + "/"

    if env:
        raw_env = bsuite.load_and_record(env, save_path, overwrite=True)
    else:
        raw_env = bsuite.load_and_record(env_config["env"], save_path, overwrite=True)
    env = gym_wrapper.GymFromDMEnv(raw_env)
    env = TransformObservation(env, lambda obs: obs.squeeze())
    return env
