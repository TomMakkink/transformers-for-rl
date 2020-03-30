""" REINFORCE algorithm on the CartPole-v1 environment. 

This implementation of REINFORCE is based on the opensource pytorch version 
which can be found here: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py. 

The agent in the CartPole enviornment is tasked with balancing a cart 
for as long as possible by moving left or right. For each timestep 
that the pole is up, a reward of +1 is provided. The observation space 
consists of 4 inputs: cart position, cart velocity, pole angle, pole velocity 
at tip.

The following implementation will experiment with the use of transformers in this algorithm. 
"""

import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from transformers.transformer_wrapper import Transformer
from ray import tune



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

env = gym.make('CartPole-v1')
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
        # self.transformer_type = config["transformer"]
        self.transformer = Transformer(
            transformer_type=config["transformer_type"],
            dim_model=config["dim_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dim_mlp=config["dim_mlp"],
            dim_head=config["dim_head"],
        )

        # if config["transformer"] == "vanilla":
        #     print("Using Transformer...")
        #     self.transformer = TransformerModel(
        #         dim_model=config["dim_model"], 
        #         num_heads=config["num_heads"], 
        #         num_encoder_layers=config["num_layers"],
        #         num_decoder_layers=1, 
        #         dim_mlp=config["dim_mlp"], 
        #         dropout=config["dropout"],
        #     )
        # elif config["transformer"] == "xl":
        #     print("Using Transformer-XL")
        #     self.transformer = TransformerXL(
        #         dim_model=config["dim_model"],
        #         num_layers=config["num_layers"],
        #         num_heads=config["num_heads"],
        #         dim_mlp=config["dim_mlp"],
        #         dim_head=config["dim_head"],
        #         mem_len=4, 
        #     )
        
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

        self.mems = tuple()

    def forward(self, x):
        # if self.transformer_type == "xl":
        #     ret = self.transformer(x, *self.mems)
        #     x, self.mems = ret[0], ret[1:]
        #     # print(f"x: {x}")
        #     # print(f"mems: {self.mems}")
        # elif self.transformer_type == "vanilla":
        #     x = self.transformer(x)
        # x = x.view(x.size(0), x.size(2))
        x = self.transformer(x)
        # x = x.view(x.size(0), x.size(2))
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


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
    policy = Policy(config)
    optimizer = optim.Adam(policy.parameters(), lr=config["lr"])
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
        # tune.track.log(episode_reward_mean=ep_reward)
        # tune.track.log(running_reward=running_reward)
        finish_episode(policy, optimizer, eps)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

        # Added for graphing 
        if i_episode > 1000:
            break; 



def main():
    # analysis = tune.run(
    #                 train, 
    #                 config = {"dim_model": 4, "num_heads": 1, "num_layers": 1, "dim_mlp": 10, "dropout": 0.0, 
    #                         "dim_head": 4, "lr":0.001, "transformer": tune.grid_search(["", "vanilla", "xl"])}
    #             )

    
    # Transformer-XL 
    config = {"dim_model": 4, "num_heads": 1, "num_layers": 1, "dim_mlp": 20, "dropout": 0.1, 
              "dim_head": 4, "lr":0.001, "transformer_type": ""}

    train(config)



if __name__ == '__main__':
    main()