import torch
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from algorithms.replay_buffer import ReplayBuffer
from utils.utils import log_to_comet_ml


class PPO():
    """
    PPO actor-critic algorithm. 
    """

    def __init__(
            self,
            name,
            actor_critic,
            env,
            steps_per_epoch,
            batch_size,
            device,
            lr,
            gamma=0.99,
            clip_ratio=0.2,
            train_iters=80,
            lam=0.97,
            ent_coef=0.0,
            value_coef=0.5,
            target_kl=0.01,
            max_ep_len=500,
            # save_freq=10,
            # image_pad=0,
            experiment=None
    ):
        """
        Args: 
        """
        super(PPO, self).__init__()
        self.ac = actor_critic(env.observation_space, env.action_space).to(device)
        self.env = env
        self.replay_buffer = ReplayBuffer(device)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.lam = lam
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.total_episodes = 0
        self.total_time_steps = 0
        self.max_ep_len = max_ep_len

        self.optimizer = Adam(self.ac.parameters(), lr=lr)

        self.writer = SummaryWriter("runs/" + name)
        self.experiment = experiment

    def collect_rollouts(self):
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        episode_returns, episode_lengths = [], []
        obs_buf, actions_buf, rewards_buf, values_buf, logp_buf = [], [], [], [], []
        for t in range(self.steps_per_epoch):
            self.total_time_steps += 1
            action, value, logp = self.ac.select_action(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device))
            next_obs, reward, done, _ = self.env.step(action)
            ep_ret += reward
            ep_len += 1
            # save
            obs_buf.append(obs)
            actions_buf.append(action)
            rewards_buf.append(reward)
            values_buf.append(value)
            logp_buf.append(logp)

            # Update obs 
            obs = next_obs

            timeout = ep_len == self.max_ep_len
            terminal = done or timeout
            epoch_ended = t == self.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        f'Warning: trajectory cut off by epoch at {ep_len} steps.')
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value, _ = self.ac.select_action(torch.as_tensor(
                        obs, dtype=torch.float32, device=self.device))
                else:
                    value = 0
                self.replay_buffer.store(obs_buf, actions_buf, rewards_buf, values_buf, logp_buf, value)
                if terminal:
                    # only save Episode Returns / Episode Length if trajectory finished
                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                    self.replay_buffer.create_new_epi()  # TODO: Check with Shahil if this is the right place

                # Reset the episode and buffers
                obs, ep_ret, ep_len = self.env.reset(), 0, 0
                obs_buf, actions_buf, rewards_buf, values_buf, logp_buf = [], [], [], [], []
                self.total_episodes += 1

        mean_episode_returns = sum(episode_returns) / len(episode_returns)
        mean_episode_length = sum(episode_lengths) / len(episode_lengths)

        return mean_episode_returns, mean_episode_length

        # for t in range(self.steps_per_epoch):
        #     self.total_time_steps += 1
        #     action, value, logp = self.ac.select_action(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        #     next_obs, reward, done, _ = self.env.step(action)
        #     ep_ret += reward
        #     ep_len += 1

        #     # save
        #     obs_buf.append(obs)
        #     actions_buf.append(action)
        #     rewards_buf.append(reward)
        #     values_buf.append(value)
        #     logp_buf.append(logp)
        #     # self.replay_buffer.store(obs, action, reward, value, logp)

        #     # Update obs 
        #     obs = next_obs

        #     timeout = ep_len == self.max_ep_len
        #     terminal = done or timeout
        #     epoch_ended = t == self.steps_per_epoch - 1

        #     if terminal or epoch_ended:
        #         if epoch_ended and not(terminal):
        #             print(f'Warning: trajectory cut off by epoch at {ep_len} steps.')
        #         # if trajectory didn't reach terminal state, bootstrap value target
        #         if timeout or epoch_ended:
        #             _, value, _ = self.ac.select_action(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        #         else:
        #             value = 0
        #         # self.replay_buffer.finish_path(value)
        #         self.replay_buffer.store(obs_buf, actions_buf, rewards_buf, values_buf, logp)
        #         if terminal:
        #             # only save EpRet / EpLen if trajectory finished
        #             episode_returns.append(ep_ret)
        #             episode_lengths.append(ep_len)
        #         obs, ep_ret, ep_len = self.env.reset(), 0, 0

        # mean_episode_returns = sum(episode_returns)/len(episode_returns)
        # mean_episode_length = sum(episode_lengths)/len(episode_lengths)

        # return mean_episode_returns, mean_episode_length

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
        replay_buffer_size = self.replay_buffer.size()
        # Train with multiple steps of gradient descent 
        for i in range(self.train_iters):
            # Sample episodes sequentially
            for episode_index in range(replay_buffer_size):
                obs, act, ret, adv, logp_old = self.replay_buffer.get(episode_index)
                # print(f"shape of obs: {obs.shape}")
                self.optimizer.zero_grad()
                action, value, logp, ent = self.ac(obs, act)
                loss_actor, kl, clipfrac = self._compute_loss_actor(logp, logp_old, adv)
                loss_critic = self._compute_loss_critic(value, ret)
                loss = loss_actor + ent * self.ent_coef + loss_critic * self.value_coef
                loss.backward()
                # plot_grad_flow(self.ac.named_parameters())
                self.optimizer.step()
                if kl > 1.5 * self.target_kl:
                    print('Early stopping at step %d due to reaching max kl.' % i)
                    break

        return loss_actor, loss_critic, loss, ent, kl

        # obs, act, ret, adv, logp_old = self.replay_buffer.get()

        # # Train with multiple steps of gradient descent 
        # for i in range(self.train_iters):
        #     for index in BatchSampler(SubsetRandomSampler(range(self.steps_per_epoch)), self.batch_size, False):
        #         self.optimizer.zero_grad()
        #         action, value, logp, ent = self.ac(obs[index], act[index])
        #         loss_actor, kl, clipfrac = self._compute_loss_actor(logp, logp_old[index], adv[index])
        #         loss_critic = self._compute_loss_critic(value, ret[index])
        #         if kl > 1.5 * self.target_kl:
        #             print('Early stopping at step %d due to reaching max kl.'%i)
        #             break
        #         loss = loss_actor + ent * self.ent_coef + loss_critic * self.value_coef
        #         loss.backward()
        #         # plot_grad_flow(self.ac.named_parameters())
        #         self.optimizer.step()

        # return loss_actor, loss_critic, loss, ent, kl

    def log_to_tensorboard(self, mean_episode_returns, mean_episode_length, loss_actor, loss_critic, loss, ent, kl):
        self.writer.add_scalar('Mean Episode Reward',
                               mean_episode_returns, self.total_time_steps)
        self.writer.add_scalar('Mean Episode Length',
                               mean_episode_length, self.total_time_steps)
        self.writer.add_scalar('Loss/Loss', loss, self.total_time_steps)
        self.writer.add_scalar(
            'Loss/Actor Loss', loss_actor, self.total_time_steps)
        self.writer.add_scalar(
            'Loss/Critic Loss', loss_critic, self.total_time_steps)
        self.writer.add_scalar('Extra/Kl', kl, self.total_time_steps)
        self.writer.add_scalar('Extra/Entropy', ent, self.total_time_steps)

    def log_to_screen(self, mean_episode_returns, mean_episode_length, loss_actor, loss_critic, loss, ent, kl):
        print("------------------------------")
        print(f"Mean episode returns: {mean_episode_returns:.2f}")
        print(f"Mean episode length: {mean_episode_length}")
        print(f"Loss actor: {loss_actor:.2f}")
        print(f"Loss critic: {loss_critic:.2f}")
        print(f"Loss: {loss:.2f}")
        print(f"Ent: {ent:.2f}")
        print(f"KL: {kl:.2f}")
        print(f"Timestep: {self.total_time_steps}")
        print(f"Episodes: {self.total_episodes}")
        print("------------------------------")

    def learn(self, total_timesteps):
        while self.total_time_steps < total_timesteps:
            mean_episode_returns, mean_episode_length = self.collect_rollouts()
            loss_actor, loss_critic, loss, ent, kl = self.train()
            self.log_to_screen(mean_episode_returns, mean_episode_length,
                               loss_actor, loss_critic, loss, ent, kl)
            self.log_to_tensorboard(
                mean_episode_returns, mean_episode_length, loss_actor, loss_critic, loss, ent, kl)
            log_to_comet_ml(self.experiment, self.total_time_steps, mean_episode_returns, mean_episode_length,
                            loss_actor, loss_critic, loss, ent, kl)
