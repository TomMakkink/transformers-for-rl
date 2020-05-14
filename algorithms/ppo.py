import torch
from algorithms.replay_buffer import ReplayBuffer
from torch.optim import Adam
from utils.general import plot_grad_flow
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPO():
    """
    PPO actor-critic algorithm, whereby the actor and critic networks are optimised seperately. 
    """
    def __init__(
        self, 
        actor_critic, 
        observation_space, 
        action_space,
        buffer_size, 
        gamma=0.99, 
        clip_ratio=0.2,
        actor_lr=3e-4,
        critic_lr=1e-3,
        train_actor_iters=80, 
        train_critic_iters=80, 
        lam=0.97, 
        target_kl=0.01, 
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

        self.actor_optimizer = Adam(self.ac.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.ac.critic.parameters(), lr=critic_lr)

        self.gamma = gamma 
        self.clip_ratio = clip_ratio
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.lam = lam 
        self.target_kl = target_kl

        self.device = device


    def store(self, obs, act, rew, val, logp):
        self.replay_buffer.store(obs, act, rew, val, logp)


    def finish_path(self, value):
        self.replay_buffer.finish_path(value)


    def step(self, obs):
        return self.ac.step(obs)


    def _compute_loss_actor(self, obs, act, adv, logp_old):
        # Policy loss
        action_dist, logp = self.ac.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = action_dist.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        action_dist_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_actor, action_dist_info


    # Set up function for computing value loss
    def _compute_loss_critic(self, obs, ret):
        return ((self.ac.critic(obs) - ret)**2).mean()


    def update(self):
        obs, act, ret, adv, logp_old = self.replay_buffer.get()
        
        logp_old = torch.as_tensor(logp_old, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        actor_l_old, actor_info_old = self._compute_loss_actor(obs, act, adv, logp_old)
        actor_l_old = actor_l_old.item()
        critic_l_old = self._compute_loss_critic(obs, ret).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            loss_actor, actor_dist_info = self._compute_loss_actor(obs, act, adv, logp_old)
            kl = actor_dist_info['kl']
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_actor.backward()
            self.actor_optimizer.step()

        # Value function learning
        for i in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()
            loss_critic = self._compute_loss_critic(obs, ret)
            loss_critic.backward()
            self.critic_optimizer.step()

        # Log changes from update
        kl, ent, cf = actor_dist_info['kl'], actor_info_old['ent'], actor_dist_info['cf']
        return loss_actor, loss_critic, kl, ent, cf
