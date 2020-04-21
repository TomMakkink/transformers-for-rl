""" 
REINFORCE algorithm on the CartPole-v1 environment. 

This implementation of REINFORCE is based on the opensource pytorch version 
which can be found here: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py. 

The agent in the CartPole enviornment is tasked with balancing a pole on a cart 
for as long as possible by moving left or right. For each timestep 
that the pole is up, a reward of +1 is provided. The observation space 
consists of 4 inputs: cart position, cart velocity, pole angle, pole velocity 
at tip.

The following implementation will experiment with the use of transformers in this algorithm. 
"""

import argparse
import gym
import numpy as np
# import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from transformers.transformer_wrapper import Transformer
from torch.utils.tensorboard import SummaryWriter
from optimisers.nadam import Nadam
import torchvision.transforms as T
from gym.wrappers import Monitor

# =====================================
# Parse command-line arguments
# =====================================

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--transformer', action="store",
                    choices=['vanilla', 'xl'],
                    help="use a transformer to 'preprocess' input")
args = parser.parse_args()

# =====================================
# Create Environment 
# =====================================

env = gym.make('CarRacing-v0')
env = Monitor(env, './video')
env.seed(args.seed)
torch.manual_seed(args.seed)

# =====================================
# Visualise Gradients 
# =====================================

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    print(f"Average grads: {ave_grads}")


# =====================================
# Policy Class
# =====================================

class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        self.transformer_type = config["transformer_type"]
        self.transformer = Transformer(
            transformer_type=config["transformer_type"],
            d_model=config["d_model"],
            output_dim=config["output_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dim_mlp=config["dim_mlp"],
            mem_len=config["mem_len"],
        )
        
        self.affine1 = nn.Linear(96 * 96 * 3, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.affine2 = nn.Linear(128, 3)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.saved_log_probs = []
        self.rewards = []

        self.saved_state = []

    def forward(self, x):
        # self.saved_state.append(x)
        # if len(self.saved_state) == 5 and self.transformer.transformer is not None:
        #     x = torch.stack(self.saved_state)
        #     x = self.transformer(x)
        #     x = x.view(1, x.size(0))
        #     self.saved_state = self.saved_state[1:]
        x = self.affine1(x.flatten())
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        # print(f"Action scores: {action_scores}")
        return action_scores


def select_action(policy, state):
    state = torch.from_numpy(state.copy()).float().unsqueeze(0)
    mu = policy(state).detach().numpy()
    std = torch.exp(policy.log_std)
    dist = Normal(mu, std)
    



def finish_episode(policy, optimizer, eps):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    # plot_grad_flow(policy.named_parameters())
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train(config): 
    writer = SummaryWriter("runs/" + config["transformer_type"])
    policy = Policy(config)
    optimizer = Nadam(policy.parameters(), lr=config["lr"])
    eps = np.finfo(np.float32).eps.item()
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(policy, state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        writer.add_scalar('Running Reward', running_reward, i_episode)
        finish_episode(policy, optimizer, eps)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

        if i_episode > 100:
            break; 



def main():
    # analysis = tune.run(
    #                 train, 
    #                 config = {"d_model": 4, "num_heads": 1, "num_layers": 1, "dim_mlp": 24, "dropout": 0.0, "output_dim": 4,
    #                         "dim_head": 4, "lr":0.001, "transformer_type": tune.grid_search(["gtrxl", "vanilla", "xl", "rezero", "none"])}
    #             )

    config = {"d_model": 4, "output_dim": 4, "num_heads": 1, "num_layers": 1, "dim_mlp": 20, "dropout": 0.0, 
              "lr":0.001, "mem_len": 2, "transformer_type": "ReZero"}

    train(config)


if __name__ == '__main__':
    main()