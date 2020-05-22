import numpy as np
import torch

from examples.pendulum import main

import gym

from stable_baselines.ppo.ppo import PPO
from stable_baselines.ppo.policy import CnnPolicy, MlpPolicy
# from stable_baselines3.ppo import CNNPolicy
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import FrameStack

if __name__ == '__main__':
    #  main(name="mountain_card/ppo_test", epochs=250, seed=543)
    seed = 543
    env = make_vec_env('CarRacing-v0', n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    # env = gym.make('Pendulum-v0')
    vf = [256, 64]
    pi = [256, 64]
    policy_kwargs = dict(net_arch=[64, 256, dict(vf=vf), dict(pi=pi)])
    model = PPO("Carracing/Transformer_10", CnnPolicy, env, verbose=1, seed=seed, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=100000)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    # print(f"Mean Reward: {mean_reward} ({std_reward})")
