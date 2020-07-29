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

        self.optimiser = optim.Adam(self.net.parameters(), lr=a2c_config["lr"])

        self.writer = SummaryWriter("runs/" + name)
        self.logger = logger
        if self.logger:
            logger.log_parameters(a2c_config)

    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            )
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[
                rand_ids, :
            ], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(
        ppo_epochs,
        mini_batch_size,
        states,
        actions,
        log_probs,
        returns,
        advantages,
        clip_param=0.2,
    ):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(
                mini_batch_size, states, actions, log_probs, returns, advantages
            ):
                dist, value = model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def learn(self, total_timesteps):

        scores = []
        scores_deque = deque(maxlen=print_every)
        episode = 1
        total_t = 0
        # for episode in range(1, number_episodes + 1):
        while total_t < total_timesteps:
            log_probs = []
            values = []
            states = []
            rewards = []
            actions = []

            state = self.env.reset()
            for t in range(a2c_config["steps_per_epoch"]):
                total_t = total_t + 1
                state = torch.from_numpy(state).float().to(self.device)
                dist, value = self.net(state)

                action = dist.sample()

                log_prob = dist.log_prob(action)

                next_state, reward, done, _ = self.env.step(action.item())

                rewards.append(reward)
                log_probs.append(log_prob.unsqueeze(0))
                values.append(value)
                actions.append(action)
                states

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
                print(
                    "Episode {}\tAverage Score: {:.2f}".format(
                        episode, np.mean(scores_deque)
                    )
                )

            metrics = {
                "Episode Return": scores[-1],
                "Episode Length": episode_length,
                "Loss/Actor Loss": policy_loss.detach().cpu().numpy(),
                "Loss/Critic Loss": value_function_loss.detach().cpu().numpy(),
                "Loss/Loss": loss.detach().cpu().numpy(),
            }
            if self.logger:
                log_to_comet_ml(self.logger, metrics, step=episode)
