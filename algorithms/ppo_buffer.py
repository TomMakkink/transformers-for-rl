# PPO buffer implementation derived from openai ppo: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
import torch 
import numpy as np
from utils.general import count_vars, combined_shape, discount_cumsum, plot_grad_flow

class PPOBuffer():
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, device, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        # rews = torch.cat((self.rew_buf[path_slice], last_val))
        # vals = torch.cat((self.val_buf[path_slice], last_val))
        
        # # GAE-Lambda advantage calculation
        # deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # deltas = deltas.cpu().detach().numpy()
        # rews = rews.cpu().detach().numpy()

        # # TODO: run this on the gpu as a torch tensor. This is definitely not the most efficient way. 
        # discounted_adv = discount_cumsum(deltas, self.gamma * self.lam)
        # self.adv_buf[path_slice] = torch.as_tensor(discounted_adv.copy(), dtype=torch.float32)
        
        # # the next line computes rewards-to-go, to be targets for the value function
        # discounted_ret = discount_cumsum(rews, self.gamma)[:-1]
        # self.ret_buf[path_slice] = torch.as_tensor(discounted_ret.copy(), dtype=torch.float32)
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = self.adv_buf.mean()
        adv_std = self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return self.obs_buf, self.act_buf, self.ret_buf, self.adv_buf, self.logp_buf

    def to(self, device):
        """
        Copy to the relevant device. 
        """
        self.obs_buf = self.obs_buf.to(device)
        # self.act_buf = self.act_buf.to(device)
        # self.adv_buf = self.adv_buf.to(device)
        # self.rew_buf = self.rew_buf.to(device)
        # self.ret_buf = self.ret_buf.to(device)
        # self.val_buf = self.val_buf.to(device)
        # self.logp_buf = self.logp_buf.to(device)
    