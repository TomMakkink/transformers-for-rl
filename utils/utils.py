import numpy as np
import scipy.signal
import torch
from environment import (
    SlidingWindowEnv,
    CustomMemoryChain,
    load_custom_memory_env,
    custom_memory_sweep,
)

import bsuite
from agents.a2c import A2C
from agents.dqn import DQN

from bsuite.utils import gym_wrapper, wrappers
from bsuite.logging.csv_logging import Logger
from bsuite import sweep
from configs.a2c_config import a2c_config
from configs.env_config import env_config
from configs.experiment_config import experiment_config
from configs.transformer_config import transformer_config
import seaborn as sns
import matplotlib.pyplot as plt


def update_configs(args):
    experiment_config.update(
        {
            "project_name": args.project,
            "experiment_name": args.name,
            "agent": args.agent,
            "memory": args.memory,
            "seed": args.seed,
        }
    )
    env_config.update({"env": args.env})
    transformer_config.update(
        {
            "max_seq_len": args.window,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dim_mlp": args.dim_mlp
        }
    )
    a2c_config.update(
        {
            'use_norm': not args.no_norm
        }
    )


def get_agent(agent_name: str):
    return {"a2c": A2C, "dqn": DQN}.get(agent_name, A2C)  # Defaults to A2C


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
    print(f"using {device}")
    experiment_config["device"] = device


def get_sweep_from_bsuite_id(bsuite_id: str):
    return {
        "umbrella_length": sweep.UMBRELLA_LENGTH,
        "umbrella_distract": sweep.UMBRELLA_DISTRACT,
        "memory_length": sweep.MEMORY_LEN,
        "memory_size": sweep.MEMORY_SIZE,
        "cartpole": sweep.CARTPOLE,
        "memory_custom": custom_memory_sweep,
    }.get(bsuite_id, [bsuite_id])


def get_save_path(window_size, agent, memory):
    return "results/" + f"{window_size}/{agent}/{memory}/"


def create_environment(agent, seed, memory, env, window_size):
    # build folder path to save data
    save_path = get_save_path(window_size, agent, memory)

    if env.startswith("memory_custom"):
        print(f"Running environment {env}")
        raw_env = load_custom_memory_env(env)
        logger = Logger(env, save_path, overwrite=True)
        raw_env = wrappers.Logging(raw_env, logger)
    elif env:
        raw_env = bsuite.load_and_record(env, save_path, overwrite=True)
    else:
        raw_env = bsuite.load_and_record(env_config["env"], save_path, overwrite=True)

    env = gym_wrapper.GymFromDMEnv(raw_env)
    env = SlidingWindowEnv(env, window_size=window_size)
    return env
