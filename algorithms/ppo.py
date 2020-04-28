import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import gym
from gym.wrappers import Monitor, FrameStack
import time
import datetime
from utils.general import count_vars, plot_grad_flow
from algorithms.ppo_buffer import PPOBuffer
from models.mlp_actor_critic import MLPActorCritic

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Summary Writer 
writer = SummaryWriter("runs/" + str(datetime.datetime.now()))


def make_env(env_name="CarRacing-v0", max_ep_len=1000, num_stack=1):
    env = gym.make(env_name)
    env._max_episode_steps = max_ep_len
    env = Monitor(env, './video', force=True)
    env = FrameStack(env, num_stack=num_stack)
    return env


def process_frames(obs):
    # Convert LazyFrame to np array. Shape: [Frames, Height, Width, Channels]
    obs = np.array(obs, copy=False)
    # Convert np array to torch tensor. 
    state = torch.as_tensor(obs, dtype=torch.float32, device=device)
    # Below may be more efficient? 
    # state = torch.from_numpy(obs).float().to(device)
    # Convert to channels first format. Shape: [Frame, Channels, Height, Width]
    state = state.permute(0, 3, 1, 2)
    state /= 255
    return state


def select_action(model, state):
    action, value, logp = model.step(state)
    # Only action, value, logp of last frame 
    # TODO: Double check that 0 is the right index
    action, value, logp = action[0], value[0], logp[0]
    action = np.tanh(action)
    action = (action + np.array([0., 1., 1.])) / np.array([1., 2., 2.])
    return action, value, logp


def compute_loss_actor(model, obs, act, adv, logp):
    # Policy loss
    logp_old = logp
    batch_size, seq_len, channels, height, width = obs.shape
    obs = obs.view((batch_size * seq_len), channels, height, width)
    actor, logp = model.actor(obs, act) 
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = actor.entropy().mean().item()
    clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    actor_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_actor, actor_info


# Set up function for computing value loss
def compute_loss_critic(model, obs, ret):
    return ((model.critic(obs) - ret)**2).mean()


def ppo_update(model, buf, iters, optimizer, ent_coef, value_coef):
    print("Updating...")
    obs, act, ret, adv, logp = buf.get()

    # Train policy with multiple steps of gradient descent
    for i in range(iters):
        for index in BatchSampler(SubsetRandomSampler(range(len(obs))), 128, False):
            optimizer.zero_grad()
            loss_actor, actor_info = compute_loss_actor(model, obs[index], act[index], adv[index], logp[index])
            loss_critic = compute_loss_critic(model, obs[index], ret[index])
            loss = loss_actor - actor_info["ent"] * ent_coef + loss_critic * value_coef
            loss.backward()
            # plot_grad_flow(ac.actor.named_parameters())
            loss.step()


def ppo(
        env_name, 
        actor_critic=MLPActorCritic, 
        seed=3, 
        steps_per_epoch=1000,
        epochs=2000, 
        gamma=0.99, 
        clip_ratio=0.2,
        lr=1e-3, 
        train_ppo_iters=80, 
        lam=0.97, 
        max_ep_len=1000,
        ent_coef=0.01, 
        value_coef=0.5, 
        save_freq=10, 
        num_stack=1, 
        repeat_action=1
    ):
        # Make environment
        env = make_env(env_name, max_ep_len, num_stack)

        # Random seeds for reproducibility 
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create actor-critic model 
        # TODO: Create an environment wrapper to output observations in (channel, height, width) rather than hardcode 
        obs_shape = (4, 3, 96, 96)
        action_shape = env.action_space.shape[0]
        ac = actor_critic(obs_shape, action_shape).to(device)

        # Set up experience buffer.
        steps_per_epoch = int(steps_per_epoch) 
        buf = PPOBuffer(obs_shape, action_shape, steps_per_epoch, gamma, lam)
        buf.to(device)

        # Set up Optimiser 
        optimizer = Adam(ac.parameters(), lr=lr)

        # Prepare for interaction with environment
        start_time = time.time()
        obs, ep_ret, ep_len = env.reset(), 0, 0
        state = process_frames(obs)

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            episode_rewards = []
            episode_lengths = []
            for t in range(steps_per_epoch):
                # Only update action every {repeat_action} number of steps 
                if t % repeat_action == 0: 
                    action, value, logp = select_action(model=ac, state=state)

                next_obs, reward, done, _ = env.step(action)
                ep_ret += reward
                ep_len += 1

                # Log to the buffer 
                buf.store(state, action, reward, value, logp)
                
                # Update state
                state = process_frames(next_obs)
                
                timeout = ep_len == max_ep_len
                terminal = done or timeout
                epoch_ended = t == steps_per_epoch-1

                if terminal or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        print("Episode did not end in terminal state.")
                        _, value, _ = select_action(ac, state)
                    else:
                        value = 0
                    buf.finish_path(device, value)
                    if terminal: 
                        episode_rewards.append(ep_ret)
                        episode_lengths.append(ep_len)
                        env.stats_recorder.save_complete()
                        env.stats_recorder.done = True
                    # TODO: Decide what to do here. e.g. mask? Break out of loop? Or just restart? 
                    obs, ep_ret, ep_len = env.reset(), 0, 0
                    state = process_frames(obs)

            # Perform PPO update
            ppo_update(ac, buf, train_ppo_iters, optimizer, ent_coef, value_coef)

            # Track mean episode return per epoch 
            mean_episode_reward = sum(episode_rewards)/len(episode_rewards)
            mean_episode_length = sum(episode_lengths)/len(episode_lengths)
            writer.add_scalar('Mean Episode Reward', mean_episode_reward, epoch)
            writer.add_scalar('Mean Episode Length', mean_episode_length, epoch)
        
        writer.close()
        env.monitor.close()


def ray_ppo(config):
    ppo(**config)
