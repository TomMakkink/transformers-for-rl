import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, max_num_episodes):
        self._storage = deque(maxlen=max_num_episodes)
        self._storage.append([])
        self.current_episode_index = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)
        self._storage[self.current_episode_index].append(data)

    # def _encode_sample(self, indices):
    #     states, actions, rewards, next_states, dones = [], [], [], [], []
    #     for i in indices:
    #         data = self._storage[i]
    #         state, action, reward, next_state, done = data
    #         states.append(state)
    #         actions.append(action)
    #         rewards.append(reward)
    #         next_states.append(next_state)
    #         dones.append(done)
    #     return (
    #         states,
    #         np.array(actions),
    #         np.array(rewards),
    #         next_states,
    #         np.array(dones),
    #     )
    def sample_random_timesteps(self, batch_size):
        flattened_experience = []
        for episode in self._storage:
            for timestep in range(len(episode)):
                flattened_experience.append(episode[timestep])
        states, actions, rewards, next_states, dones = zip(
            *random.sample(flattened_experience, batch_size)
        )
        return (
            states,
            np.array(actions),
            np.array(rewards),
            next_states,
            np.array(dones),
        )

    def sample_random_episode(self):
        episode_index = np.random.randint(0, len(self._storage))
        states, actions, rewards, next_states, dones = zip(
            *self._storage[episode_index]
        )
        return (
            states,
            np.array(actions),
            np.array(rewards),
            next_states,
            np.array(dones),
        )

    def sample(self, batch_size, device, sample_sequentially=False):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        if sample_sequentially:
            states, actions, rewards, next_states, dones = self.sample_random_episode()
        else:
            states, actions, rewards, next_states, dones = self.sample_random_timesteps(
                batch_size
            )
        # states, actions, rewards, next_states, dones = self.sample_random_episode()
        # else:
        #     print(len(self._storage))
        # states, actions, rewards, next_states, dones = self._encode_sample()

        states = torch.stack(states).to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.from_numpy(dones).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def reset(self):
        if self.current_episode_index + 1 < self._storage.maxlen:
            self.current_episode_index += 1
        self._storage.append([])
