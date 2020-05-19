import datetime
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from algorithms.ppo_single_loss import PPO 

# GPU or CPU
def train(name, env_fn=None, actor_critic=None, device="cpu", seed=0,
        steps_per_epoch=5000, epochs=50, gamma=0.99, clip_ratio=0.2, lr=3e-4,
        train_iters=80, lam=0.97, max_ep_len=500, target_kl=0.01, save_freq=10, frame_stack=0):
    
    # Summary Writer
    writer = SummaryWriter("runs/pendulum/" + name)

    # Create the environment 
    env = env_fn(env_name='Pendulum-v0', seed=seed)

    agent = PPO(actor_critic, env.observation_space, env.action_space, buffer_size=steps_per_epoch, device=device)

    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        episode_returns = []
        episode_lengths = []
        for t in range(steps_per_epoch):
            a, v, logp = agent.select_action(torch.as_tensor(obs, dtype=torch.float32))
            print(logp)
            next_obs, r, d, _ = env.step(a * np.array([4] - np.array([2])))
            ep_ret += r
            ep_len += 1

            # save and log
            agent.store(obs, a, r, v, logp)
            
            # Update obs 
            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = agent.select_action(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                agent.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                obs, ep_ret, ep_len = env.reset(), 0, 0

        # Perform PPO update!
        loss_actor, loss_critic, loss, ent, kl = agent.update()

        print("Updated")

        # Track mean episode return per epoch 
        mean_episode_returns = sum(episode_returns)/len(episode_returns)
        mean_episode_length = sum(episode_lengths)/len(episode_lengths)
        writer.add_scalar('Mean Episode Length', mean_episode_length, epoch)
        writer.add_scalar('Actor Loss', loss_actor, epoch)
        writer.add_scalar('Critic Loss', loss_critic, epoch)
        writer.add_scalar('Kl', kl, epoch)
        writer.add_scalar('Entropy', ent, epoch)
        writer.add_scalar('Mean Episode Reward', mean_episode_returns, epoch) 
        writer.add_scalar('Loss', loss, epoch)