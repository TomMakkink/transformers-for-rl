from collections import deque
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from typing import Deque, Dict, List, Tuple
from configs.ppo_config import ppo_config
from configs.env_config import env_config
import numpy as np
from models.actor_critic_mlp import Actor, Critic
from utils.logging import log_to_comet_ml, log_to_screen


class PPO:
    """
    PPO actor-critic algorithm. 
    """

    def __init__(self, name, model, env, device, logger):
        """
        Args: 
        """
        super(PPO, self).__init__()

        self.env = env
        self.device = device
        self.memory = Memory()

        self.gamma = ppo_config['gamma']
        self.eps_clip = ppo_config['eps_clip']
        self.epochs = ppo_config['epochs']
        self.update_timestep = ppo_config['update_timestep']
        self.learning_rate = ppo_config['learning_rate']
        self.log_interval = ppo_config['log_interval']
        self.entropy_weight = ppo_config['entropy_weight']
        # network
        # TODO: Use model that's passed in
        hidden_layers_size = [64]
        self.actor = Actor(env.observation_space, env.action_space,
                           hidden_layers_size).to(self.device)
        self.critic = Critic(env.observation_space, hidden_layers_size).to(self.device)

        self.old_actor = Actor(env.observation_space, env.action_space,
                               hidden_layers_size).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=self.learning_rate * 5)

        self.MseLoss = nn.MSELoss()

        self.writer = SummaryWriter("runs/" + name)
        self.logger = logger

    def update(self):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards),
                                       reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        actor_losses_l, critic_losses_l = [], []
        for _ in range(self.epochs):
            # Evaluating old actions and values :
            dist = self.actor(old_states)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(old_states)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values

            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2) - (self.entropy_weight * dist_entropy)
            actor_loss = actor_loss.mean()

            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_losses_l.append(actor_loss.item())
            critic_losses_l.append(critic_loss.item())

        actor_loss_l = sum(actor_losses_l) / len(actor_losses_l)
        critic_loss_l = sum(critic_losses_l) / len(critic_losses_l)

        # Copy new weights into old policy:
        self.old_actor.load_state_dict(self.actor.state_dict())
        return actor_loss_l, critic_loss_l

    def learn(self, total_episodes, window_size=1):
        solved_reward = 230

        # logging variables
        running_reward = 0
        avg_length = 0
        timestep = 0

        self.memory.clear_memory()

        # training loop
        for i_episode in range(1, total_episodes + 1):
            state = self.env.reset()
            for t in range(env_config['max_episode_length']):
                timestep += 1

                # Running policy_old:
                state = torch.from_numpy(state).float().to(self.device)
                dist = self.old_actor(state)
                action = dist.sample()
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.logprobs.append(dist.log_prob(action))

                state, reward, done, _ = self.env.step(action.item())

                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                # update if its time
                if timestep % self.update_timestep == 0:
                    actor_loss, critic_loss = self.update()
                    metrics = {
                        "Actor Loss": actor_loss,
                        "Critic Loss": critic_loss
                    }
                    if self.logger:
                        log_to_comet_ml(self.logger, metrics, step=i_episode)
                    metrics.update({"Episode": i_episode})
                    log_to_screen(metrics)
                    self.memory.clear_memory()
                    timestep = 0

                running_reward += reward
                if done:
                    break

            avg_length += t

            # # stop training if avg_reward > solved_reward
            # if running_reward > (self.log_interval * solved_reward):
            #     print("########## Solved! ##########")
            #     break

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))
                metrics = {
                    "Average Score": running_reward
                }
                if self.logger:
                    log_to_comet_ml(self.logger, metrics, step=i_episode)
                metrics.update({"Episode": i_episode})
                log_to_screen(metrics)

                running_reward = 0
                avg_length = 0


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
