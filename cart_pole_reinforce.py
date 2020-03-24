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
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from transformers.transformer import TransformerModel
from transformers.transformer_xl import TransformerXL
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
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)

# =====================================
# Policy Class
# =====================================

class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        if args.transformer == "vanilla":
            print("Using Transformer...")
            self.transformer = TransformerModel(
                d_model=config["d_model"], 
                n_head=config["n_heads"], 
                num_encoder_layers=1,
                num_decoder_layers=0, 
                dim_feedforward=config["dim_feedforward"], 
                dropout=config["dropout"],
            )
        elif args.transformer == "xl":
            print("Using Transformer-XL")
            # self, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int,
            #      resid_p:float=0., attn_p:float=0., ff_p:float=0., embed_p:float=0., bias:bool=False, scale:bool=True,
            #      mask:bool=True, mem_len:int=0):
            self.transformer = TransformerXL(
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                d_inner=config["dim_feedforward"],
                d_head=config["d_head"],
                mem_len=0, 
            )
        
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        if args.transformer == "xl":
            mems = []
            p = self.transformer(x, *mems)
        
        x = self.affine1(x.view(1,4))
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
        finish_episode(policy, optimizer, eps)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            # plot_grad_flow(policy.named_parameters())
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


def main():
    # analysis = tune.run(
    # train, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

    # print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))
    
    config = {"d_model": 4, "n_heads": 1, "n_layers": 1, "dim_feedforward": 10, "dropout": 0.0, 
              "d_head": 4, "lr":0.001}

    train(config)



if __name__ == '__main__':
    main()