from omegaconf import DictConfig
import hydra
from environment import SlidingWindowEnv, load_custom_memory_env

import bsuite
from bsuite.utils import gym_wrapper, wrappers
from bsuite.logging.csv_logging import Logger


def build_agent(
    state_size: int,
    action_size: int,
    agent_cfg: DictConfig,
    memory_cfg: DictConfig,
    device,
):
    if memory_cfg._target_ is not None:
        memory = hydra.utils.instantiate(
            memory_cfg,
            input_dim=agent_cfg.hidden_size[-1],
            output_dim=agent_cfg.hidden_size[-1],
        )
    else:
        memory = None

    agent = hydra.utils.instantiate(
        agent_cfg,
        state_size=state_size,
        action_size=action_size,
        memory=memory,
        device=device,
    )

    return agent


def build_env(env, window, device, save_dir):
    if env.startswith("memory_custom"):
        print(f"Running environment {env}")
        raw_env = load_custom_memory_env(env)
        logger = Logger(env, save_dir, overwrite=True)
        raw_env = wrappers.Logging(raw_env, logger)
    else:
        raw_env = bsuite.load_and_record(env, save_dir, overwrite=True)

    env = gym_wrapper.GymFromDMEnv(raw_env)
    env = SlidingWindowEnv(env, window_size=window, device=device)
    return env
