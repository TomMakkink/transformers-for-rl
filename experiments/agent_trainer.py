from configs.experiment_config import experiment_config
from utils.logging import log_to_screen, log_to_comet_ml
from collections import deque
import torch
import numpy as np

from utils.visualisation import viz_attention


def train_agent(agent, env, total_episodes=10000, logger=None, viz=False):
    scores = []
    scores_deque = deque(maxlen=experiment_config["log_interval"])
    loss_deque = deque(maxlen=experiment_config["log_interval"])

    if logger is not None:
        logger.log_parameters(experiment_config)
    
    total_step = 0
    max_steps = 6 # note this is hardcoded
    total_steps = total_episodes*max_steps
    avg_attend = [0]
    
    for episode in range(1, total_episodes + 1):
        rewards = []
        state = env.reset()
        
        
        for t in range(experiment_config["max_steps_per_episode"]):
            action = agent.act(state.unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            total_step += 1
            
            
            if viz == True:
                weights = agent.net.attention_weights
                correct_attend = viz_attention(weights.detach().numpy(), episode, t, total_step, 
                            avg_attend, total_steps)  
                mean_attend = int(correct_attend)/total_step # I think this is wrong
                avg_attend.append(mean_attend)
            
            agent.collect_experience(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        loss = agent.optimize_network()
        agent.reset()

        episode_length = len(rewards)
        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))
        loss_deque.append(loss)

        if episode % experiment_config["log_interval"] == 0:
            metrics = {
                "Average Score": np.mean(scores_deque),
                "Loss": np.mean(loss_deque),
            }
            if logger:
                log_to_comet_ml(logger, metrics, step=episode)
            metrics.update({"Episode": episode})
            log_to_screen(metrics)
