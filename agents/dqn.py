import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.mlp import MLP
from agents.agent import Agent
from configs.dqn_config import dqn_config
from configs.experiment_config import experiment_config
from agents.replay_buffer import ReplayBuffer
import math
import random


class DQN(Agent):
    def __init__(self, state_size, action_size, memory):
        super(DQN, self).__init__(state_size, action_size, memory)
        self.device = experiment_config["device"]
        self.net = MLP(state_size, action_size, memory_type=memory).to(self.device)
        self.replay_buffer = ReplayBuffer(dqn_config["buffer_size"])
        self.optimiser = optim.Adam(self.net.parameters(), lr=dqn_config["lr"])
        self.current_timestep = 1
        self.action_size = action_size
        self.sample_sequentially = (
            True if self.net.memory_network.memory is not None else False
        )

    def act(self, state):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        epsilon = self.calculate_epsilon(self.current_timestep)
        self.current_timestep += 1
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.net(state)
                _, action = q_values.max(1)
                return action.item()
        else:
            return random.randrange(self.action_size)

    def calculate_epsilon(self, current_timestep):
        return dqn_config["epsilon"]["final"] + (
            dqn_config["epsilon"]["start"] - dqn_config["epsilon"]["final"]
        ) * math.exp(-1.0 * current_timestep / dqn_config["epsilon"]["decay"])

    def optimize_network(self):
        self.net.reset()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            dqn_config["batch_size"], self.device, self.sample_sequentially
        )

        next_q_values = self.net(next_states)
        max_next_q_values, _ = next_q_values.max(1)
        target_q_values = (
            rewards + (1 - dones) * dqn_config["gamma"] * max_next_q_values
        )

        input_q_values = self.net(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss

    def reset(self):
        self.replay_buffer.reset()
        self.net.reset()

    def collect_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

