from configs.experiment_config import experiment_config
from utils.logging import log_to_screen
from collections import deque
import torch
import numpy as np


def train_agent(agent, env, total_episodes, device, logger=None, window_size=1):
    scores = []
    scores_deque = deque(maxlen=experiment_config["log_interval"])
    loss_deque = deque(maxlen=experiment_config["log_interval"])
    obs_window = deque(maxlen=window_size)
    if logger:
        logger.log_parameters(config)

    for episode in range(1, total_episodes + 1):
        rewards = []
        state = env.reset()

        for t in range(experiment_config["max_steps_per_episode"]):
            state = torch.from_numpy(state).float().to(device)
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            agent.collect_experience(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        loss = agent.optimize_network()
        agent.reset()

        episode_length = len(rewards)
        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))
        loss_deque.append(loss.item())

        if episode % experiment_config["log_interval"] == 0:
            metrics = {
                "Average Score": np.mean(scores_deque),
                "Loss": np.mean(loss_deque),
            }
            if logger:
                log_to_comet_ml(logger, metrics, step=episode)
            metrics.update({"Episode": episode})
            log_to_screen(metrics)
