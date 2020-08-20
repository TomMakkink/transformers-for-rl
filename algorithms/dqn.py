# DQN Implementation adapted from: https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
import math
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from configs.dqn_config import dqn_config
from algorithms.replay_buffer import ReplayBuffer
from utils.logging import log_to_screen, log_to_comet_ml


class DQN:
    def __init__(self, name, model, env, device, logger):
        self.device = device
        self.env = env
        self.net = model(env.observation_space, env.action_space).to(self.device)
        self.replay_buffer = ReplayBuffer(dqn_config["max_steps_per_episode"])

        self.optimiser = optim.Adam(self.net.parameters(), lr=dqn_config["lr"])

        self.writer = SummaryWriter("runs/" + name)
        self.logger = logger
        if self.logger:
            logger.log_parameters(dqn_config)

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.from_numpy(np.float32(state))
        next_state = torch.from_numpy(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.net(state)
        next_q_values = self.net(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + dqn_config["gamma"] * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.data).pow(2).mean()

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def calculate_epsilon(self, current_timestep):
        return dqn_config["epsilon"]["final"] + (
            dqn_config["epsilon"]["start"] - dqn_config["epsilon"]["final"]
        ) * math.exp(-1.0 * current_timestep / dqn_config["epsilon"]["decay"])

    def learn(self, total_episodes, window_size=1):
        rewards_deque = deque(maxlen=dqn_config["log_interval"])
        loss_deque = deque(maxlen=dqn_config["log_interval"])
        current_timestep = 1

        for episode in range(1, total_episodes + 1):
            episode_reward = 0
            state = self.env.reset()

            for t in range(dqn_config["max_steps_per_episode"]):
                epsilon = self.calculate_epsilon(current_timestep)
                action = self.net.act(state, epsilon)

                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                current_timestep += 1

                if len(self.replay_buffer) > dqn_config["batch_size"]:
                    loss = self.compute_td_loss(dqn_config["batch_size"])
                    loss_deque.append(loss.item())

                if done:
                    rewards_deque.append(episode_reward)
                    break

            if episode % dqn_config["log_interval"] == 0:
                metrics = {
                    "Average Score": np.mean(rewards_deque),
                    "Loss": np.mean(loss_deque),
                }
                if self.logger:
                    log_to_comet_ml(self.logger, metrics, step=episode)
                metrics.update({"Episode": episode})
                log_to_screen(metrics)

        # current_timestep = 1
        # losses = []
        # all_rewards = []
        # episode_reward = 0

        # state = self.env.reset()

        # for episode in range(1, total_episodes + 1):
        #     epsilon = self.calculate_epsilon(current_timestep)
        #     action = self.net.act(state, epsilon)

        #     next_state, reward, done, _ = self.env.step(action)
        #     self.replay_buffer.push(state, action, reward, next_state, done)

        #     state = next_state
        #     episode_reward += reward
        #     current_timestep += 1

        #     if done:
        #         state = self.env.reset()
        #         all_rewards.append(episode_reward)
        #         episode_reward = 0

        #     if len(self.replay_buffer) > dqn_config["batch_size"]:
        #         loss = self.compute_td_loss(dqn_config["batch_size"])
        #         losses.append(loss.item())

        #     if episode % dqn_config["log_interval"] == 0:
        #         metrics = {
        #             "Average Score": np.mean(all_rewards),
        #             "Loss": np.mean(losses),
        #         }
        #         # if self.logger:
        #         #     log_to_comet_ml(self.logger, metrics, step=episode)
        #         metrics.update({"Episode": episode})
        #         log_to_screen(metrics)

