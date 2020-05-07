import datetime
import time

import numpy as np

import gym
import torch
from algorithms.replay_buffer import ReplayBuffer
from gym.wrappers import FrameStack, Monitor, ResizeObservation
from optimisers.larc import LARC
from torch.distributions import Beta
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from utils.general import count_vars, plot_grad_flow

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Summary Writer 
writer = SummaryWriter("runs/" + str(datetime.datetime.now()))


class PPO():
    """
    PPO actor-critic algorithm
    """
    def __init__(
        self, 
        actor_critic_model, 
        obs_shape, 
        action_shape, 
        buffer_size=1000,
        gamma=0.99, 
        clip_ratio=0.2,
        lr=1e-3, 
        train_ppo_iters=10, 
        lam=0.97, 
        ent_coef=0.01, 
        value_coef=0.5, 
        save_freq=10, 
        image_pad=4,
    ):
        """
        Args: 
        """
        super(PPO, self).__init__()
        self.actor_critic = actor_critic_model(obs_shape, action_shape).to(device)
        self.clip_ratio = clip_ratio
        self.train_ppo_iters = train_ppo_iters
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.save_freq = save_freq

        self.optimizer = Adam(self.actor_critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(obs_shape, action_shape, buffer_size, image_pad, device, gamma, lam)

    def select_action(self, state):
        with torch.no_grad(): 
            alpha, beta, value = self.actor_critic(state)
        dist = Beta(alpha, beta)
        action = dist.sample() 
        logp = dist.log_prob(action).sum(dim=1)
        logp = logp.item()

        action = action.squeeze().cpu().numpy()[0]
        value = value.squeeze().cpu().numpy()[0]
        # Shift action to be approprite for car racing observation space 
        action = action * np.array([2., 1., 1.]) + np.array([-1, 0., 0.])
        return action, value, logp

    def store(self, obs, act, rew, val, logp):
        self.replay_buffer.store(obs, act, rew, val, logp)

    def finish_path(self, value):
        self.replay_buffer.finish_path(value)

    def compute_loss_actor(self, adv, logp, logp_old):
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        actor_info = dict(kl=approx_kl, cf=clipfrac)

        return loss_actor, actor_info

    def compute_loss_critic(self, value, ret):
        return ((value - ret)**2).mean()

    def update(self):
        print("Updating...")
        obs_buf, act, ret, adv, logp = self.replay_buffer.get()
        
        # TODO: Need to improve numpy -> torch tensor and GPU -> CPU. 
        # TODO: Standardies whether its obs or state
        logp_buf = torch.as_tensor(logp, dtype=torch.float32, device=device)
        adv_buf = torch.as_tensor(adv, dtype=torch.float32, device=device)
        act_buf = torch.as_tensor(act, dtype=torch.float32, device=device)
        ret_buf = torch.as_tensor(ret, dtype=torch.float32, device=device)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_ppo_iters):
            for index in BatchSampler(SequentialSampler(range(len(obs_buf))), 250, False):
                self.optimizer.zero_grad()
                obs, act, adv, logp_old, ret = obs_buf[index], act_buf[index], adv_buf[index], logp_buf[index], ret_buf[index]
                alpha, beta, value = self.actor_critic(obs)
                dist = Beta(alpha, beta)
                action = dist.sample() 
                ent = dist.entropy().mean().item()
                logp = dist.log_prob(action).sum(dim=1, keepdim=True)

                loss_actor, actor_info = self.compute_loss_actor(adv, logp, logp_old)
                loss_critic = self.compute_loss_critic(value, ret)
                loss = loss_actor - ent * self.ent_coef + loss_critic * self.value_coef
                loss.backward()
                self.optimizer.step()
        
        return loss, loss_actor, loss_critic, actor_info["kl"], ent


def make_env(env_name="CarRacing-v0", max_ep_len=1000, num_stack=1, seed=43, monitor=True):
    env = gym.make(env_name)
    env._max_episode_steps = max_ep_len
    if monitor: env = Monitor(env, './video', force=True)
    env = ResizeObservation(env, 64)
    env = FrameStack(env, num_stack=num_stack)
    env.seed(seed)
    return env

def process_frame(obs):
    # Convert LazyFrame to np array. Shape: [Frames, Height, Width, Channels]
    obs = np.array(obs, copy=False)
    # Convert np array to torch tensor. 
    state = torch.as_tensor(obs, dtype=torch.float32, device=device)
    # Convert to channels first format. Shape: [Frame, Channels, Height, Width]
    state = state.permute(0, 3, 1, 2)
    state /= 255
    return state.unsqueeze(0)


def train(epochs, steps_per_epoch, repeat_action, seed, ppo_config, env_config):
    # Make environment
    env = make_env(seed=seed, **env_config)

    # TODO: don't hardcode the obs or action space. Create carracing wrapper.  
    agent = PPO(obs_shape=(20, 3, 64, 64), action_shape=3, buffer_size=steps_per_epoch, **ppo_config)

    # Prepare for interaction with environment
    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0

    state = process_frame(obs)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        episode_rewards = []
        episode_lengths = []
        for t in range(steps_per_epoch):
            # Only update action every {repeat_action} number of steps 
            if t % repeat_action == 0: 
                action, value, logp = agent.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            ep_ret += reward
            ep_len += 1

            # Log to the buffer 
            agent.store(state, action, reward, value, logp)
            
            # Update state
            state = process_frame(next_obs)
            
            timeout = ep_len == env_config['max_ep_len']
            terminal = done or timeout 
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    print("Episode did not end in terminal state.")
                    _, value, _ = agent.select_action(state)
                else:
                    value = 0
                agent.finish_path(value)
                if terminal: 
                    print("Episode ended in terminal state.")
                    episode_rewards.append(ep_ret)
                    episode_lengths.append(ep_len)
                env.stats_recorder.save_complete()
                env.stats_recorder.done = True
                obs, ep_ret, ep_len = env.reset(), 0, 0
                state = process_frame(obs)

        # Perform PPO update
        loss, loss_actor, loss_critic, kl, ent = agent.update()

        # Track mean episode return per epoch 
        mean_episode_reward = sum(episode_rewards)/len(episode_rewards)
        mean_episode_length = sum(episode_lengths)/len(episode_lengths)
        writer.add_scalar('Mean Episode Length', mean_episode_length, epoch)
        writer.add_scalar('Actor Loss', loss_actor, epoch)
        writer.add_scalar('Critic Loss', loss_critic, epoch)
        writer.add_scalar('Kl', kl, epoch)
        writer.add_scalar('Mean Episode Reward', mean_episode_reward, epoch) 
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Entropy', ent, epoch)
        
    
    writer.close()
    # env.monitor.close()
