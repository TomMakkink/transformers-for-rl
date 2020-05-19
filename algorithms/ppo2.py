import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from algorithms.replay_buffer import ReplayBuffer

class PPO2():
    """
    PPO actor-critic algorithm. 
    """
    def __init__(
        self, 
        name, 
        actor_critic, 
        env, 
        steps_per_epoch, 
        batch_size, 
        device, 
        lr=0.001,              
        gamma=0.99, 
        clip_ratio=0.2,
        train_iters=80, 
        lam=0.97, 
        ent_coef=0.0, 
        value_coef=0.5, 
        # max_ep_len=500, 
        # save_freq=10, 
        # image_pad=0,
    ):
        """
        Args: 
        """
        super(PPO2, self).__init__()
        self.ac = actor_critic(env.observation_space, env.action_space).to(device)
        self.env = env 
        self.replay_buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape, 
                                        steps_per_epoch, 0, device, gamma, lam)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.device = device 
        self.gamma = gamma 
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.lam = lam 
        self.ent_coef = ent_coef
        self.value_coef = value_coef

        self.optimizer = Adam(self.ac.parameters(), lr=lr)

        self.writer = SummaryWriter("runs/" + name)

 
    def collect_rollouts(self):
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        episode_returns = []
        episode_lengths = []
        for t in range(self.steps_per_epoch):
            action, value, logp = self.ac.select_action(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, done, _ = self.env.step(action * np.array([4] - np.array([2])))
            ep_ret += reward
            ep_len += 1

            # save and log
            self.replay_buffer.store(obs, action, reward, value, logp)
            
            # Update obs 
            obs = next_obs

            timeout = ep_len == self.env._max_episode_steps
            terminal = done or timeout
            epoch_ended = t == self.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps.')
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value, _ = self.ac.select_action(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    value = 0
                self.replay_buffer.finish_path(value)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                obs, ep_ret, ep_len = self.env.reset(), 0, 0
        
        mean_episode_returns = sum(episode_returns)/len(episode_returns)
        mean_episode_length = sum(episode_lengths)/len(episode_lengths)
        
        return mean_episode_returns, mean_episode_length

 
    def _compute_loss_actor(self, logp, logp_old, adv):
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return loss_actor, approx_kl, clipfrac


    def _compute_loss_critic(self, value, ret):
        return ((value - ret)**2).mean()


    def train(self):
        obs, act, ret, adv, logp_old = self.replay_buffer.get()

        # Train with multiple steps of gradient descent 
        for i in range(self.train_iters):
            self.optimizer.zero_grad()
            action, value, logp, ent = self.ac(obs, act)
            loss_actor, kl, clipfrac = self._compute_loss_actor(logp, logp_old, adv)
            loss_critic = self._compute_loss_critic(value, ret).item()
            loss = loss_actor + ent * self.ent_coef + loss_critic * self.value_coef
            loss.backward()
            self.optimizer.step()

        return loss_actor, loss_critic, loss, ent, kl

    
    def log_to_tensorboard(self, mean_episode_returns, mean_episode_length, loss_actor, loss_critic, loss, ent, kl, epoch):
        self.writer.add_scalar('Mean Episode Reward', mean_episode_returns, epoch) 
        self.writer.add_scalar('Loss', loss, epoch)
        self.writer.add_scalar('Mean Episode Length', mean_episode_length, epoch)
        self.writer.add_scalar('Actor Loss', loss_actor, epoch)
        self.writer.add_scalar('Critic Loss', loss_critic, epoch)
        self.writer.add_scalar('Kl', kl, epoch)
        self.writer.add_scalar('Entropy', ent, epoch)



    # May implement total timesteps later on
    def learn(self, epochs):  
        for epoch in range(epochs): 
            mean_episode_returns, mean_episode_length = self.collect_rollouts()
            loss_actor, loss_critic, loss, ent, kl = self.train()
            self.log_to_tensorboard(mean_episode_returns, mean_episode_length, loss_actor, loss_critic, loss, ent, kl, epoch)

    

