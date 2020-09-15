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
    def __init__(self, state_size, action_size, hidden_size, memory):
        super(DQN, self).__init__(state_size, action_size, hidden_size, memory)
        self.device = experiment_config["device"]
        self.policy_net = MLP(
            state_size, action_size, hidden_size, memory_type=memory
        ).to(self.device)
        self.target_network = MLP(
            state_size, action_size, hidden_size, memory_type=memory
        ).to(self.device)
        self.update_target_update_by_percentage()
        self.target_network.eval()
        self.replay_buffer = ReplayBuffer(dqn_config["buffer_size"])
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=dqn_config["lr"])
        self.current_timestep = 1
        self.action_size = action_size
        self.sample_sequentially = (
            True if self.policy_net.memory_network.memory is not None else False
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
                q_values = self.policy_net(state)
                _, action = q_values.max(1)
                return action.item()
        else:
            return random.randrange(self.action_size)

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_net.state_dict())

    def update_target_update_by_percentage(self):
        for param, target_param in zip(
            self.policy_net.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                dqn_config["lr"] * param.data
                + (1 - dqn_config["lr"]) * target_param.data
            )

    def calculate_epsilon(self, current_timestep):
        return dqn_config["epsilon"]["final"] + (
            dqn_config["epsilon"]["start"] - dqn_config["epsilon"]["final"]
        ) * math.exp(-1.0 * current_timestep / dqn_config["epsilon"]["decay"])

    def optimize_network(self):
        self.policy_net.reset()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            dqn_config["batch_size"], self.device, self.sample_sequentially
        )

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values, _ = next_q_values.max(1)
            target_q_values = (
                rewards + (1 - dones) * dqn_config["gamma"] * max_next_q_values
            )

        input_q_values = self.policy_net(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.update_target_update_by_percentage()

        return loss.item()

    def reset(self):
        self.replay_buffer.reset()
        self.policy_net.reset()
        self.target_network.reset()

    def collect_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

