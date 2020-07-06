# PPO buffer implementation derived from openai ppo: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
from collections import deque
import torch 
import torch.nn as nn
# import kornia
import numpy as np
from utils.utils import count_vars, combined_shape, discount_cumsum, plot_grad_flow

class ReplayBuffer(): 
    """
    A buffer for storing complete episodes experienced by a PPO agents, 
    and using Generalizaed Advantage Estimation (GAE-Lambda) to calculate 
    the advantages of state-action pairs. 
    """
    def __init__(self, device, max_epi_num=50, gamma=0.99, lam=0.95):
        self.max_epi_num = max_epi_num
        self.memory = deque(maxlen=max_epi_num)
        self.memory.append([])
        self.current_epi = 0 
        self.is_av = False 
        self.gamma = gamma
        self.lam = lam
        self.device = device
    
    def reset(self):
        self.current_epi = 0
        self.memory.clear()
        self.memory.append([])

    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def store(self, obs, actions, rewards, values, logp, last_val):
        # The "last_val" argument should be 0 if the trajectory ended
        # because the agent reached a terminal state (died), and otherwise
        # should be V(s_T), the value function estimated for the last state.
        # This allows us to bootstrap the reward-to-go calculation to account
        # for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        # Explicitly convert to numpy array 
        actions = np.array(actions, dtype=np.float32)
        logp = np.array(logp, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)

        rewards = np.append(rewards, last_val)
        values = np.append(values, last_val)

        # Call at the end of an episode
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.lam)

        # the next two lines implement the advantage normalization trick (shifted to have
        # mean zero and std one).
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / adv_std

        # the next line computes rewards-to-go, to be targets for the value function
        returns = discount_cumsum(rewards, self.gamma)[:-1]
        
        self.memory[self.current_epi].append([obs, actions, returns, advantages, logp])


    def get(self, episode_index):
        """
        Call this at the end of an epoch to get all of the data from the buffer. 
        """
        episode = self.memory[episode_index]
        obs, actions, returns, advantages, logp = episode[0][0], episode[0][1], episode[0][2], episode[0][3], episode[0][4]
        
        # Convert from numpy arrays to pytorch tensors
        print(f"Obs before conversion: {obs}") 
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logp = torch.as_tensor(logp, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns.copy(), dtype=torch.float32, device=self.device)

        return obs, actions, returns, advantages, logp

    def size(self):
        return len(self.memory)



# class ReplayBuffer():
#     """
#     A buffer for storing trajectories experienced by a PPO agent interacting
#     with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
#     for calculating the advantages of state-action pairs.
#     """
#     def __init__(self, obs_dim, act_dim, size, image_pad, device, gamma=0.99, lam=0.95):
#         self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
#         self.adv_buf = np.zeros(size, dtype=np.float32)
#         self.rew_buf = np.zeros(size, dtype=np.float32)
#         self.ret_buf = np.zeros(size, dtype=np.float32)
#         self.val_buf = np.zeros(size, dtype=np.float32)
#         self.logp_buf = np.zeros(size, dtype=np.float32)

#         # self.aug_obs = image_pad > 0
#         # if self.aug_obs: self.aug_trans = nn.Sequential(
#         #                 nn.ReplicationPad2d(image_pad),
#         #                 kornia.augmentation.RandomCrop((obs_dim[-1], obs_dim[-1])))

#         self.gamma, self.lam = gamma, lam
#         self.ptr, self.path_start_idx, self.max_size = 0, 0, size
#         self.device = device

#     def store(self, obs, act, rew, val, logp):
#         """
#         Append one timestep of agent-environment interaction to the buffer.
#         """
#         assert self.ptr < self.max_size     # buffer has to have room so you can store
        
#         self.obs_buf[self.ptr] = obs
#         self.act_buf[self.ptr] = act
#         self.rew_buf[self.ptr] = rew
#         self.val_buf[self.ptr] = val
#         self.logp_buf[self.ptr] = logp
#         self.ptr += 1

#     def finish_path(self, last_val=0):
#         """
#         Call this at the end of a trajectory, or when one gets cut off
#         by an epoch ending. This looks back in the buffer to where the
#         trajectory started, and uses rewards and value estimates from
#         the whole trajectory to compute advantage estimates with GAE-Lambda,
#         as well as compute the rewards-to-go for each state, to use as
#         the targets for the value function.
#         The "last_val" argument should be 0 if the trajectory ended
#         because the agent reached a terminal state (died), and otherwise
#         should be V(s_T), the value function estimated for the last state.
#         This allows us to bootstrap the reward-to-go calculation to account
#         for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
#         """
#         path_slice = slice(self.path_start_idx, self.ptr)
#         rews = np.append(self.rew_buf[path_slice], last_val)
#         vals = np.append(self.val_buf[path_slice], last_val)

#         # the next two lines implement GAE-Lambda advantage calculation
#         deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
#         self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

#         # the next line computes rewards-to-go, to be targets for the value function
#         self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
#         self.path_start_idx = self.ptr

#     def get(self):
#         """
#         Call this at the end of an epoch to get all of the data from
#         the buffer, with advantages appropriately normalized (shifted to have
#         mean zero and std one). Also, resets some pointers in the buffer.
#         """
#         assert self.ptr == self.max_size    # buffer has to be full before you can get
#         self.ptr, self.path_start_idx = 0, 0
#         # the next two lines implement the advantage normalization trick
#         adv_mean = self.adv_buf.mean()
#         adv_std = self.adv_buf.std()
#         self.adv_buf = (self.adv_buf - adv_mean) / adv_std

#         # if self.aug_obs:
#         #     obs = obs.squeeze()
#         #     obs = self.aug_trans(obs)

#         obs = torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device)
#         logp = torch.as_tensor(self.logp_buf, dtype=torch.float32, device=self.device)
#         adv = torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device)
#         act = torch.as_tensor(self.act_buf, dtype=torch.float32, device=self.device)
#         ret = torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device)

#         return obs, act, ret, adv, logp

    