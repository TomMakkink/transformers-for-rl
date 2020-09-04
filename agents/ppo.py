from agents.agent import Agent
from models.actor_critic_mlp import ActorCriticMLP
from configs.ppo_config import ppo_config
from configs.experiment_config import experiment_config
import numpy as np
import torch
import torch.optim as optim


class PPO(Agent):
    def __init__(self, state_size, action_size, memory, hidden_size):
        super(PPO, self).__init__(state_size, action_size, memory, hidden_size)
        self.device = experiment_config["device"]
        self.net = ActorCriticMLP(state_size, action_size, hidden_size, memory).to(
            self.device
        )
        self.optimiser = optim.Adam(self.net.parameters(), lr=ppo_config["lr"])
        self.clip_ratio = ppo_config["clip_ratio"]
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []

    def _compute_returns(self):
        R = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + ppo_config["gamma"] * R
            returns.insert(0, R)
        returns = np.array(returns)
        returns -= returns.mean()
        if returns.std() > 0.0:
            returns /= returns.std()
        return returns

    def optimize_network(self):
        self.net.reset()
        returns = self._compute_returns()
        returns = torch.from_numpy(returns).float().to(self.device).detach()

        states = torch.stack(self.states)
        values = torch.cat(self.values).detach()
        actions = torch.tensor(self.actions)
        old_log_probs = torch.cat(self.log_probs).detach()
        advantage = returns - values

        for _ in range(ppo_config["epochs"]):
            dist, values = self.net(states)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantage
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        return loss

    def reset(self):
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.net.reset()

    def act(self, state):
        dist, value = self.net(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.log_probs.append(log_prob)
        self.values.append(value)

        return action.item()

    def collect_experience(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)

