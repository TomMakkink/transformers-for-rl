import datetime
import time
import numpy as np
import torch
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from env.carracing import make_env
from algorithms.ppo import PPO


# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Summary Writer 
writer = SummaryWriter("runs/" + str(datetime.datetime.now()))


def select_action(model, obs):
    with torch.no_grad(): 
        alpha, beta, value = model(obs)
    dist = Beta(alpha, beta)
    action = dist.sample() 
    logp = dist.log_prob(action).sum(dim=1)
    logp = logp.item()

    action = action.squeeze().cpu().numpy()[0]
    value = value.squeeze().cpu().numpy()[0]
    # Shift action to be approprite for car racing observation space 
    action = action * np.array([2., 1., 1.]) + np.array([-1, 0., 0.])
    return action, value, logp


def distribution(model, obs):
    alpha, beta, value = model(obs)
    dist = Beta(alpha, beta)
    action = dist.sample() 
    logp = dist.log_prob(action).sum(dim=1, keepdim=True)
    return action, value, logp


def train(epochs, steps_per_epoch, repeat_action, seed, ppo_config, env_config):
    # Make environment
    env = make_env(seed=seed, **env_config, device=device)

    # TODO: Still not ideal. Think of a way to improve this. 
    frames, height, width, channel = env.observation_space.shape
    obs_shape = (frames, channel, height, width)
    action_shape = env.action_space.shape[0]
    agent = PPO(obs_shape=obs_shape, action_shape=action_shape, buffer_size=steps_per_epoch, **ppo_config, device=device)

    # Prepare for interaction with environment
    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        episode_returns = []
        episode_lengths = []
        last_100_rewards = np.zeros(100)
        count = 0 
        for t in range(steps_per_epoch):
            # Only update action every {repeat_action} number of steps 
            if t % repeat_action == 0: 
                action, value, logp = select_action(agent.actor_critic, obs)
            next_obs, reward, done, _ = env.step(action)
            last_100_rewards[(count % 100)] = reward 
            ep_ret += reward
            ep_len += 1
            count += 1

            # Log to the buffer 
            agent.store(obs, action, reward, value, logp)
            
            # Update state
            obs = next_obs
            
            timeout = ep_len == env_config['max_ep_len']
            if last_100_rewards.mean() <= -0.1: done = True
            terminal = done or timeout 
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    print("Episode did not end in terminal state.")
                    _, value, _ = select_action(agent.actor_critic, obs)
                else:
                    value = 0
                agent.finish_path(value)
                if terminal: 
                    print("Episode ended in terminal state.")
                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                env.stats_recorder.save_complete()
                env.stats_recorder.done = True
                obs, ep_ret, ep_len = env.reset(), 0, 0

        # Perform PPO update
        # loss, loss_actor, loss_critic, kl, ent = agent.update()
        loss, loss_actor, loss_critic, kl = agent.update()


        # Track mean episode return per epoch 
        mean_episode_returns = sum(episode_returns)/len(episode_returns)
        mean_episode_length = sum(episode_lengths)/len(episode_lengths)
        writer.add_scalar('Mean Episode Length', mean_episode_length, epoch)
        writer.add_scalar('Actor Loss', loss_actor, epoch)
        writer.add_scalar('Critic Loss', loss_critic, epoch)
        writer.add_scalar('Kl', kl, epoch)
        writer.add_scalar('Mean Episode Reward', mean_episode_returns, epoch) 
        writer.add_scalar('Loss', loss, epoch)
        # writer.add_scalar('Entropy', ent, epoch)
        
    
    writer.close()
    # env.monitor.close()