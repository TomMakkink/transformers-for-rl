from baselines.common.vec_env import dummy_vec_env
from baselines.ppo2 import ppo2
from baselines.bench import Monitor
import bsuite
from bsuite.utils import gym_wrapper
import tensorflow as tf
from utils.utils import set_random_seed, get_device

# Instructions: 
# 1: Download and install openai baselines 
#    git clone https://github.com/openai/baselines.git
#    python -m pip install baselines/
# 2: Check version of tensorflow==1.x 
# 3: For tensorboard logging test 
#    export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'
#    export OPENAI_LOGDIR=...directory...

def _load_env():
    raw_env = bsuite.load_from_id(bsuite_id='cartpole/0')
    env = gym_wrapper.GymFromDMEnv(raw_env)
    return Monitor(env, 'runs/' + "ppo_test/")

def ppo_test(name, total_timesteps, seed):
    device = get_device()
    set_random_seed(seed)
    env = dummy_vec_env.DummyVecEnv([_load_env])
    ppo2.learn(env=env, network='mlp', lr=1e-3, total_timesteps=total_timesteps)