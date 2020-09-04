from agents.agent import Agent
from models.actor_critic_mlp import ActorCriticMLP
from configs.ppo_config import ppo_config
from configs.experiment_config import experiment_config
import numpy as np
import torch
import torch.optim as optim


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p is not None:
            print(f"Layer name: {n}")
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    print(f"Average grads: {ave_grads}")


class PPO(Agent):
    """
    PPO actor-critic agent. 
    """

    def __init__(self, state_size, action_size, memory, hidden_size):
        super(PPO, self).__init__(state_size, action_size, memory, hidden_size)
        self.device = experiment_config["device"]
        self.net = ActorCriticMLP(
            state_size, action_size, hidden_size, memory_type=memory
        ).to(self.device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=ppo_config["lr"])
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.entropy = 0

    def compute_gae(self, next_value, rewards, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * tau * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[
                rand_ids
            ], advantage[rand_ids]

    def ppo_update(
        self,
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
            losses = []
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(
                mini_batch_size, states, actions, log_probs, returns, advantages
            ):
                dist, value = self.net(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                approx_kl = (old_log_probs - new_log_probs).mean().item()
                # if approx_kl > 1.5 * ppo_config["target_kl"]:
                #     # print("Early topping due to reaching max kl divergence.")
                #     break

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # losses.append(loss.item())

        # return np.mean(losses)

    def optimize_network(self):
        next_value = 0  # Assume it is done
        returns = self.compute_gae(next_value, self.rewards, self.values)

        returns = torch.stack(returns).squeeze(1).detach()
        log_probs = torch.stack(self.log_probs).detach()
        values = torch.stack(self.values).squeeze(1).detach()
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions)
        advantage = returns - values

        self.ppo_update(
            ppo_config["epochs"],
            ppo_config["mini_batch_size"],
            states,
            actions,
            log_probs,
            returns,
            advantage,
        )
        return torch.tensor(0)

    def act(self, state):
        dist, value = self.net(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.entropy += dist.entropy().mean()

        self.values.append(value)
        self.log_probs.append(log_prob)

        return action.item()

    def collect_experience(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)

    def reset(self):
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.entropy = 0
        self.net.reset()


# class PPO:
#     """
#     PPO actor-critic algorithm.
#     """

#     def __init__(self, name, model, env, device, logger):
#         """
#         Args:
#         """
#         super(PPO, self).__init__()

#         self.env = env
#         self.device = device
#         self.memory = Memory()

#         self.gamma = ppo_config["gamma"]
#         self.eps_clip = ppo_config["eps_clip"]
#         self.epochs = ppo_config["epochs"]
#         self.update_timestep = ppo_config["update_timestep"]
#         self.learning_rate = ppo_config["learning_rate"]
#         self.log_interval = ppo_config["log_interval"]
#         self.entropy_weight = ppo_config["entropy_weight"]
#         # network
#         # TODO: Use model that's passed in
#         hidden_layers_size = [64]
#         self.actor = Actor(
#             env.observation_space, env.action_space, hidden_layers_size
#         ).to(self.device)
#         self.critic = Critic(env.observation_space, hidden_layers_size).to(self.device)

#         self.old_actor = Actor(
#             env.observation_space, env.action_space, hidden_layers_size
#         ).to(self.device)
#         self.old_actor.load_state_dict(self.actor.state_dict())

#         # optimizer
#         self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
#         self.critic_optimizer = Adam(
#             self.critic.parameters(), lr=self.learning_rate * 5
#         )

#         self.MseLoss = nn.MSELoss()

#         self.writer = SummaryWriter("runs/" + name)
#         self.logger = logger

#     def update(self):

#         rewards = []
#         discounted_reward = 0
#         for reward, is_terminal in zip(
#             reversed(self.memory.rewards), reversed(self.memory.is_terminals)
#         ):
#             if is_terminal:
#                 discounted_reward = 0
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)

#         # Normalizing the rewards:
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

#         # convert list to tensor
#         old_states = torch.stack(self.memory.states).to(self.device).detach()
#         old_actions = torch.stack(self.memory.actions).to(self.device).detach()
#         old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

#         # Optimize policy for K epochs:
#         actor_losses_l, critic_losses_l = [], []
#         for _ in range(self.epochs):
#             # Evaluating old actions and values :
#             dist = self.actor(old_states)
#             logprobs = dist.log_prob(old_actions)
#             dist_entropy = dist.entropy()
#             state_values = self.critic(old_states)

#             # Finding the ratio (pi_theta / pi_theta__old):
#             ratios = torch.exp(logprobs - old_logprobs.detach())

#             # Finding Surrogate Loss:
#             advantages = rewards - state_values

#             surr1 = ratios * advantages.detach()
#             surr2 = (
#                 torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
#                 * advantages.detach()
#             )
#             actor_loss = -torch.min(surr1, surr2) - (self.entropy_weight * dist_entropy)
#             actor_loss = actor_loss.mean()

#             critic_loss = 0.5 * self.MseLoss(state_values, rewards)
#             # train critic
#             self.critic_optimizer.zero_grad()
#             critic_loss.backward()
#             self.critic_optimizer.step()

#             # train actor
#             self.actor_optimizer.zero_grad()
#             actor_loss.backward()
#             self.actor_optimizer.step()
#             actor_losses_l.append(actor_loss.item())
#             critic_losses_l.append(critic_loss.item())

#         actor_loss_l = sum(actor_losses_l) / len(actor_losses_l)
#         critic_loss_l = sum(critic_losses_l) / len(critic_losses_l)

#         # Copy new weights into old policy:
#         self.old_actor.load_state_dict(self.actor.state_dict())
#         return actor_loss_l, critic_loss_l

#     def learn(self, total_episodes, window_size=1):
#         solved_reward = 230

#         # logging variables
#         running_reward = 0
#         avg_length = 0
#         timestep = 0

#         self.memory.clear_memory()

#         # training loop
#         for i_episode in range(1, total_episodes + 1):
#             state = self.env.reset()
#             for t in range(env_config["max_episode_length"]):
#                 timestep += 1

#                 # Running policy_old:
#                 state = torch.from_numpy(state).float().to(self.device)
#                 dist = self.old_actor(state)
#                 action = dist.sample()
#                 self.memory.states.append(state)
#                 self.memory.actions.append(action)
#                 self.memory.logprobs.append(dist.log_prob(action))

#                 state, reward, done, _ = self.env.step(action.item())

#                 # Saving reward and is_terminal:
#                 self.memory.rewards.append(reward)
#                 self.memory.is_terminals.append(done)

#                 # update if its time
#                 if timestep % self.update_timestep == 0:
#                     actor_loss, critic_loss = self.update()
#                     metrics = {"Actor Loss": actor_loss, "Critic Loss": critic_loss}
#                     if self.logger:
#                         log_to_comet_ml(self.logger, metrics, step=i_episode)
#                     metrics.update({"Episode": i_episode})
#                     log_to_screen(metrics)
#                     self.memory.clear_memory()
#                     timestep = 0

#                 running_reward += reward
#                 if done:
#                     break

#             avg_length += t

#             # # stop training if avg_reward > solved_reward
#             # if running_reward > (self.log_interval * solved_reward):
#             #     print("########## Solved! ##########")
#             #     break

#             # logging
#             if i_episode % self.log_interval == 0:
#                 avg_length = int(avg_length / self.log_interval)
#                 running_reward = int((running_reward / self.log_interval))
#                 metrics = {"Average Score": running_reward}
#                 if self.logger:
#                     log_to_comet_ml(self.logger, metrics, step=i_episode)
#                 metrics.update({"Episode": i_episode})
#                 log_to_screen(metrics)

#                 running_reward = 0
#                 avg_length = 0


# class Memory:
#     def __init__(self):
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards = []
#         self.is_terminals = []

#     def clear_memory(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards[:]
#         del self.is_terminals[:]
