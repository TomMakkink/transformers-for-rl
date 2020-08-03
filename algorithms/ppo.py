from collections import deque
import torch
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from algorithms.replay_buffer import ReplayBuffer
from utils.logging import log_to_comet_ml, log_to_screen, log_to_tensorboard
from utils.utils import plot_grad_flow
from configs.ppo_config import ppo_config
from configs.env_config import env_config


class PPO:
    """
    PPO actor-critic algorithm. 
    """

    def __init__(self, name, model, env, device, logger):
        """
        Args: 
        """
        super(PPO, self).__init__()
        self.ac = model(env.observation_space, env.action_space).to(device)
        self.env = env
        self.replay_buffer = ReplayBuffer(
            device, gamma=ppo_config["gamma"], lam=ppo_config["lam"]
        )
        self.steps_per_epoch = ppo_config["steps_per_epoch"]
        self.clip_ratio = ppo_config["clip_ratio"]
        self.device = device
        self.total_episodes = 0
        self.total_time_steps = 0

        self.optimizer = Adam(self.ac.parameters(), lr=ppo_config["lr"])

        self.writer = SummaryWriter("runs/" + name)
        self.logger = logger

    def collect_rollouts(self):
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        episode_returns, episode_lengths = [], []
        obs_buf, actions_buf, rewards_buf, values_buf, logp_buf = [], [], [], [], []
        for t in range(self.steps_per_epoch):
            self.total_time_steps += 1
            obs_buf.append(obs)
            action, value, logp = self.ac.select_action(
                torch.as_tensor(obs_buf, dtype=torch.float32,
                                device=self.device)
            )
            next_obs, reward, done, _ = self.env.step(action)
            ep_ret += reward
            ep_len += 1

            # save
            actions_buf.append(action)
            rewards_buf.append(reward)
            values_buf.append(value)
            logp_buf.append(logp)

            # Update obs
            obs = next_obs

            timeout = ep_len == env_config["max_episode_length"]
            terminal = done or timeout
            epoch_ended = t == self.steps_per_epoch - 1

            if terminal or epoch_ended:
                episode_returns.append(ep_ret)
                episode_lengths.append(ep_len)
                self.replay_buffer.create_new_epi()
                self.replay_buffer.store(
                    obs_buf, actions_buf, rewards_buf, values_buf, logp_buf, value
                )

                # Reset the episode and buffers
                obs, ep_ret, ep_len = self.env.reset(), 0, 0
                obs_buf, actions_buf, rewards_buf, values_buf, logp_buf = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                self.total_episodes += 1

        mean_episode_returns = sum(episode_returns) / len(episode_returns)
        mean_episode_length = sum(episode_lengths) / len(episode_lengths)

        return mean_episode_returns, mean_episode_length

    def _compute_loss_actor(self, logp, logp_old, adv):
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                               1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return loss_actor, approx_kl, clipfrac

    def _compute_loss_critic(self, value, ret):
        return ((value - ret) ** 2).mean()

    def train(self):
        # Train with multiple steps of gradient descent
        for i in range(ppo_config["train_iters"]):
            # Sample episodes sequentially - TODO: Change to random
            for episode_index in range(self.replay_buffer.size()):
                obs, act, ret, adv, logp_old = self.replay_buffer.get(
                    episode_index)
                # for index in BatchSampler(SequentialSampler(range(len(obs))), ppo_config['batch_size'], False):
                self.optimizer.zero_grad()
                action, value, logp, ent = self.ac(obs, act)
                loss_actor, kl, clipfrac = self._compute_loss_actor(
                    logp, logp_old, adv)
                loss_critic = self._compute_loss_critic(value, ret)
                loss = (
                    loss_actor
                    + ent * ppo_config["ent_coef"]
                    + loss_critic * ppo_config["value_coef"]
                )
                loss.backward()
                # plot_grad_flow(self.ac.named_parameters())
                self.optimizer.step()
            if kl > 1.5 * ppo_config["target_kl"]:
                print("Early stopping at step %d due to reaching max kl." % i)
                break

        return loss_actor, loss_critic, loss, ent, kl

    def learn(self, total_timesteps):
        while self.total_time_steps < total_timesteps:
            self.replay_buffer.reset()
            mean_episode_returns, mean_episode_length = self.collect_rollouts()
            loss_actor, loss_critic, loss, ent, kl = self.train()

            # Logging metrics
            metrics = {
                "Mean Episode Return": mean_episode_returns,
                "Mean Episode Length": mean_episode_length,
                "Loss/Actor Loss": loss_actor,
                "Loss/Critic Loss": loss_critic,
                "Loss/Loss": loss,
                "Extra/Entropy": ent,
                "Extra/KL": kl,
            }
            log_to_tensorboard(self.writer, metrics,
                               step=self.total_time_steps)
            if self.logger:
                log_to_comet_ml(self.logger, metrics,
                                step=self.total_time_steps)
            metrics.update(
                {
                    "Total Episodes": self.total_episodes,
                    "Total Timesteps": self.total_time_steps,
                }
            )
            log_to_screen(metrics)
