import torch
from algorithms.replay_buffer import ReplayBuffer
from torch.optim import Adam
from utils.utils import plot_grad_flow
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPO():
    """
    PPO actor-critic algorithm, whereby there is a single loss function. 
    """
    def __init__(
        self, 
        actor_critic, 
        observation_space, 
        action_space,
        buffer_size, 
        gamma=0.99, 
        clip_ratio=0.2,
        lr=0.001,               # 3e-4
        train_iters=80, 
        lam=0.97, 
        ent_coef=0.01, 
        value_coef=0.5, 
        save_freq=10, 
        image_pad=0,
        device="cpu",
    ):
        """
        Args:
        """
        super(PPO, self).__init__()
        self.ac = actor_critic(observation_space, action_space)

        # Get dimension for observation and action space
        obs_dim = observation_space.shape
        act_dim = action_space.shape
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, 0, device, gamma, lam)

        self.optimizer = Adam(self.ac.parameters(), lr=lr)

        self.gamma = gamma 
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.lam = lam 
        self.ent_coef = ent_coef
        self.value_coef = value_coef

        self.device = device


    def store(self, obs, act, rew, val, logp):
        self.replay_buffer.store(obs, act, rew, val, logp)


    def finish_path(self, value):
        self.replay_buffer.finish_path(value)

    
    def select_action(self, obs):
        return self.ac.select_action(obs)


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


    def update(self):
        obs, act, ret, adv, logp_old = self.replay_buffer.get()

        # Train with multiple steps of gradient descent 
        for i in range(self.train_iters):
            self.optimizer.zero_grad()
            action, value, logp, ent = self.ac(obs, act)
            loss_actor, kl, clipfrac = self._compute_loss_actor(logp, logp_old, adv)
            loss_critic = self._compute_loss_critic(value, ret)
            loss = loss_actor + ent * self.ent_coef + loss_critic * self.value_coef
            loss.backward()
            self.optimizer.step()

        return loss_actor, loss_critic, loss, ent, kl

    