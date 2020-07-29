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
from utils.logging import log_to_comet_ml, log_to_screen


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

    def _compute_returns(self, rewards):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + a2c_config["gamma"] * R
            returns.insert(0, R)
        returns = np.array(returns)
        returns -= returns.mean()
        returns /= returns.std()
        return returns

    def learn(self, total_episodes, window_size=1, log_interval=10):
        scores = []
<<<<<<< HEAD
        scores_deque = deque(maxlen=print_every)
        loss_deque = deque(maxlen=print_every)
        episode = 1
        total_t = 0
        # for episode in range(1, number_episodes + 1):
        while total_t < total_timesteps:
=======
        scores_deque = deque(maxlen=log_interval)
        loss_deque = deque(maxlen=log_interval)
        obs_window = deque(maxlen=window_size)

        for episode in range(1, total_episodes + 1):
>>>>>>> 6f01ed53452693bc639bfbfbf7c75c9ef7d8d8b6
            log_probs = []
            values = []
            rewards = []
            state = self.env.reset()
            for t in range(a2c_config["max_steps_per_episode"]):
                state = torch.from_numpy(state).float().to(self.device)    
                obs_window.append(state)
                if t == 0: 
                    for i in range(window_size - 1): 
                        obs_window.append(state)
        
                state = torch.stack(list(obs_window))

                dist, value = self.net(state)

                action = dist.sample()[-1]

                log_prob = dist.log_prob(action)[-1]

                state, reward, done, _ = self.env.step(action.item())

                rewards.append(reward)
                log_probs.append(log_prob.unsqueeze(0))
                values.append(value)

                if done: break

            if type(self.net) is TransformerA2C:
                self.net.reset_mem()

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
<<<<<<< HEAD
            loss_deque.append(loss)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if episode % print_every == 0:
                # print(
                #     "Episode {}\tAverage Score: {:.2f}".format(
                #         episode, np.mean(scores_deque)
                #     )
                # )
                metrics = {
                    "Average Score": np.mean(scores_deque),
                    "Loss": np.mean(loss_deque),
                }
                log_to_screen(metrics)
                if self.logger:
                    log_to_comet_ml(self.logger, metrics, step=episode)

            # metrics = {
            #     "Episode Return": scores[-1],
            #     "Episode Length": episode_length,
            #     "Loss/Actor Loss": policy_loss.detach().cpu().numpy(),
            #     "Loss/Critic Loss": value_function_loss.detach().cpu().numpy(),
            #     "Loss/Loss": loss.detach().cpu().numpy(),
            # }

=======
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            loss_deque.append(loss.detach().cpu().numpy())

            if episode % log_interval == 0:
                metrics = {
                    "Average Score": np.mean(scores_deque), 
                    "Loss": np.mean(loss_deque),
                }
                if self.logger:
                    log_to_comet_ml(self.logger, metrics, step=episode)
                metrics.update({"Episode": episode})
                log_to_screen(metrics)
>>>>>>> 6f01ed53452693bc639bfbfbf7c75c9ef7d8d8b6
