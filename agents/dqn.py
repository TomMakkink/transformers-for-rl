import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from agents.agent import Agent
from configs.dqn_config import dqn_config
from agents.replay_buffer import ReplayBuffer
import math
import random


class DQN(Agent):
    def __init__(self, model, env, device):
        super(DQN, self).__init__(model, env, device)
        self.net = model(env.observation_space, env.action_space).to(self.device)
        self.replay_buffer = ReplayBuffer(dqn_config["buffer_size"])
        self.optimiser = optim.Adam(self.net.parameters(), lr=dqn_config["lr"])
        self.env = env
        self.current_timestep = 1

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
                _, action = q_values.max(0)
                return action.item()
        else:
            return self.env.action_space.sample()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def calculate_epsilon(self, current_timestep):
        return dqn_config["epsilon"]["final"] + (
            dqn_config["epsilon"]["start"] - dqn_config["epsilon"]["final"]
        ) * math.exp(-1.0 * current_timestep / dqn_config["epsilon"]["decay"])

    def optimise_network(self, *args):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            dqn_config["batch_size"], self.device
        )

        # with torch.no_grad():
        #     next_q_values = self.target_network(next_states)
        #     max_next_q_values, _ = next_q_values.max(1)
        #     target_q_values = (
        #         rewards + (1 - dones) * dqn_config["gamma"] * max_next_q_values
        #     )
        next_q_values = self.net(next_states)
        max_next_q_values, _ = next_q_values.max(1)
        target_q_values = target_q_values = (
            rewards + (1 - dones) * dqn_config["gamma"] * max_next_q_values
        )

        input_q_values = self.net(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss

    def has_replay_buffer():
        return False

