""" REINFORCE algorithm on the cartpole-v1 environment. 

This implementation is based on the opensource pytorch version 
which can be found here: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py. 

This algorithm is to be used as a 'simplest possible case' policy gradient, 
so that we can begin experimenting with transformers in RL enviornments. 
"""

import gym 
import numpy as np 
from itertools import count

import torch 
import torch.nn as nn 
import torch.nn.Functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

env = gym.make('CartPole-v1')
# Hard Code the Seed for now 
env.seed(1)
torch.manual_seed(1)

class Policy(nn.Module): 
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n 
        self.affine1 = nn.Linear(self.state_space, 128) 
        self.dropout = nn.Dropout(p=0.6)
        self.addine2 = nn.Linear(128, self.action_space)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

lr = 1e-2
gamma=0.99

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item() 

def select_action(state): 
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item() 

def finish_episode(): 
    R = 0
    policy_loss = []
    rewards = []

    # Discounted future rewards 
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R 
        rewards.insert(0, R)

    # Scale rewards 
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    
    # Calculate loss 
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum() 
    
    # Update network weights 
    optimizer.zero_grad() 
    policy_loss.backward() 
    optimizer.step() 

    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    running_reward = 10 
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0 
        for t in range(1, 10000):  # Prevents infitinit loop while learning 
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            policy.rewards += rewards 
            if done: 
                break 
        
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode() 
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

if __name__ == '__main__':
    main()



    

