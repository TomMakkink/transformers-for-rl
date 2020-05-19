import datetime
import time

import torch

from algorithms.ppo2 import PPO2

def train(
        name:str, 
        env_fn, 
        actor_critic,
        seed:int, 
        device,
        steps_per_epoch=2048, 
        epochs=100,
        batch_size=64,
        lr=0.0003, 
        gamma=0.99, 
        clip_ratio=0.2, 
        train_iters=10, 
        lam=0.95, 
        # max_ep_len=500, 
        # save_freq=10, 
        # frame_stack=0
    ):
        """
        Args: 
        """
        env = env_fn(env_name='Pendulum-v0', seed=seed)
        model = PPO2(name=name, 
                    actor_critic=actor_critic, 
                    env=env, 
                    steps_per_epoch=steps_per_epoch, 
                    batch_size=64, 
                    device=device, 
                    lr=lr,              
                    gamma=gamma, 
                    clip_ratio=clip_ratio,
                    train_iters=train_iters, 
                    lam=lam, 
                    ent_coef=0.0, 
                    value_coef=0.5)

        model.learn(epochs)
    
        
