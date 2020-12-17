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

from bsuite.utils import gym_wrapper, wrappers
from bsuite.logging.csv_logging import Logger
from bsuite import sweep


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


def get_sweep_from_bsuite_id(bsuite_id: str):
    return {
        "umbrella_length": sweep.UMBRELLA_LENGTH,
        "umbrella_distract": sweep.UMBRELLA_DISTRACT,
        "memory_len": sweep.MEMORY_LEN,
        "memory_size": sweep.MEMORY_SIZE,
        "cartpole": sweep.CARTPOLE,
        "memory_custom": custom_memory_sweep,
    }.get(bsuite_id, [bsuite_id])


def get_save_path(window_size, agent, memory, eval_run=False):
    mem_str = memory
    if memory not in ["lstm", None]:
        if (transformer_config["num_heads"] > 1) or (
            transformer_config["num_layers"] > 1
        ):
            mem_str = f"{mem_str}_h{transformer_config['num_heads']}_l{transformer_config['num_layers']}"
        if transformer_config["dim_mlp"] != 32:
            mem_str = f"{mem_str}_d{transformer_config['dim_mlp']}"

    folder_path = f"{window_size}/{agent}/{mem_str}/"
    if eval_run:
        return "results/eval/" + folder_path
    return "results/" + folder_path


def create_environment(agent, memory, env, window_size, device):
    # build folder path to save data
    save_path = get_save_path(window_size, agent, memory, eval_run)

    if env.startswith("memory_custom"):
        print(f"Running environment {env}")
        raw_env = load_custom_memory_env(env)
        logger = Logger(env, save_path, overwrite=True)
        raw_env = wrappers.Logging(raw_env, logger)
    else:
        raw_env = bsuite.load_and_record(env, save_path, overwrite=True)

    env = gym_wrapper.GymFromDMEnv(raw_env)
    env = SlidingWindowEnv(env, window_size=window_size, device=device)
    return env
