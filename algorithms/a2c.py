from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from configs.a2c_config import a2c_config
import numpy as np
from models.transformer_a2c import TransformerA2C
from utils.logging import log_to_comet_ml


class A2C:

    def __init__(self, name, model, env, device, logger):
        self.device = device
        self.env = env
        self.net = model(env.observation_space, env.action_space).to(self.device)

        self.optimiser = optim.Adam(self.net.parameters(), lr=a2c_config['lr'])

        self.writer = SummaryWriter("runs/" + name)
        self.logger = logger
        if self.logger: logger.log_parameters(a2c_config)

    def _compute_returns(self, rewards):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + a2c_config['gamma'] * R
            returns.insert(0, R)
        returns = np.array(returns)
        returns -= returns.mean()
        returns /= returns.std()
        return returns

    def learn(self, total_timesteps):
        print_every = 10
        number_episodes = 1000  # total_timesteps // a2c_config['steps_per_epoch']

        scores = []
        scores_deque = deque(maxlen=print_every)
        episode = 1
        total_t = 0
        # for episode in range(1, number_episodes + 1):
        while (total_t < total_timesteps):
            log_probs = []
            values = []
            rewards = []

            state = self.env.reset()
            for t in range(a2c_config['steps_per_epoch']):
                total_t = total_t + 1
                state = torch.from_numpy(state).float().to(self.device)
                dist, value = self.net(state)

                action = dist.sample()

                log_prob = dist.log_prob(action)

                state, reward, done, _ = self.env.step(action.item())

                rewards.append(reward)
                log_probs.append(log_prob.unsqueeze(0))
                values.append(value)

                if done:
                    episode = episode + 1
                    if type(self.net) is TransformerA2C:
                        self.net.reset_mem()
                    break

            episode_length = len(rewards)
            scores.append(sum(rewards))
            scores_deque.append(sum(rewards))

            returns = self._compute_returns(rewards)
            returns = torch.from_numpy(returns).float().to(self.device)

            values = torch.cat(values)
            log_probs = torch.cat(log_probs)

            delta = returns - values

            policy_loss = -torch.sum(log_probs * delta.detach())

            value_function_loss = 0.5 * torch.sum(delta ** 2)

            loss = policy_loss + value_function_loss

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

            metrics = {
                "Episode Return": scores[-1],
                "Episode Length": episode_length,
                "Loss/Actor Loss": policy_loss,
                "Loss/Critic Loss": value_function_loss,
                "Loss/Loss": loss}
            if self.logger: log_to_comet_ml(self.logger, metrics, step=episode)
