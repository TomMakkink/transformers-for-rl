import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as transforms 
from torch.distributions import Categorical
import ray 
from ray import tune
import numpy as np
import deepmind_lab 
from torch.utils.tensorboard import SummaryWriter
from transformers.transformer_wrapper import Transformer
from nadam import Nadam
from itertools import count
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
env = deepmind_lab.Lab("contributed/dmlab30/explore_object_locations_small", ['RGB_INTERLEAVED'],
                    {'fps': '30', 'width': '96', 'height': '72'})
env.reset()

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
        
        self.linear_input = nn.Linear(72 * 96 * 3, 256)
        self.dropout = nn.Dropout(p=0.1)
        # Only 5 outputs for now, to ignore jump and crouch 
        self.linear_output = nn.Linear(256, 5)  

        self.saved_log_probs = []
        self.rewards = []

        self.saved_state = []

    def forward(self, x):
        x = x.view(-1, 72 * 96 * 3)
        x = self.linear_input(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.transformer(x)
        action_scores = self.linear_output(x)
        return F.softmax(action_scores, dim=1)



DEFAULT_ACTION_SET = [
    np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc),    # Forward
    np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.intc),   # Backward
    np.array([0, 0, -1, 0, 0, 0, 0], dtype=np.intc),   # Strafe Left
    np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.intc),    # Strafe Right
    np.array([-20, 0, 0, 0, 0, 0, 0], dtype=np.intc),  # Look Left
    np.array([20, 0, 0, 0, 0, 0, 0], dtype=np.intc),   # Look Right
    np.array([-20, 0, 0, 1, 0, 0, 0], dtype=np.intc),  # Look Left + Forward
    np.array([20, 0, 0, 1, 0, 0, 0], dtype=np.intc),   # Look Right + Forward
    np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.intc),    # Fire.
]


def select_action(policy, state):
    state = transform(state)
    state = state.to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return DEFAULT_ACTION_SET[action.item()]


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
    plot_grad_flow(policy.named_parameters())
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train(config): 
    writer = SummaryWriter("runs/" + config["transformer_type"])
    policy = Policy(config)
    policy.to(device)
    optimizer = Nadam(policy.parameters(), lr=config["lr"])
    eps = np.finfo(np.float32).eps.item()
    running_reward = 10
    saved_images = torch.zeros(100, 72, 96, 3)
    counter = 0
    for i_episode in count(1):
        env.reset()
        state, ep_reward = env.observations(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(policy, state['RGB_INTERLEAVED'])
            state = env.observations()
            reward = env.step(action, num_steps=1)
            policy.rewards.append(reward)
            ep_reward += reward
            if env.is_running() == False: 
                break
            # if t % 10 == 0:
            #     img = torch.from_numpy(state['RGB_INTERLEAVED'])
            #     saved_images[counter] = img
            #     counter = counter + 1 
        # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        writer.add_scalar('Episode Reward', ep_reward, i_episode)
        finish_episode(policy, optimizer, eps)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if running_reward > 100:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

        if i_episode > 100:
            # writer.add_images('Environment', saved_images, dataformats='NHWC')
            writer.close()
            break; 


def main():
    # tune.run(
    #     train, 
    #     config = {"d_model": 4, "output_dim": 4, "num_heads": 1, "num_layers": 1, "dim_mlp": 20, "dropout": 0.0, 
    #         "lr":0.001, "mem_len": 2, "transformer_type": "none"}
    # )
    config = {"d_model": 256, "output_dim": 256, "num_heads": 1, "num_layers": 12, "dim_mlp": 256, "dropout": 0.1, 
              "lr":0.000005, "mem_len": 0, "transformer_type": "ReZero"}

    train(config)


    # print(f"Finished after {length} episodes, total reward is: {reward}")
    
    # env.reset()
    # observation_spec = env.observation_spec()
    # print(f"Observation spec: {observation_spec}\n")

    # action_spec = env.action_spec()
    # print(f"Action spec: {action_spec}\n")





if __name__ == '__main__':
    main()