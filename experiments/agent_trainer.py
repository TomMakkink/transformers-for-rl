from configs.experiment_config import experiment_config
from utils.logging import log_to_screen, log_to_comet_ml
from collections import deque
import numpy as np


def train_agent(agent, env, total_episodes=10000, logger=None):
    scores = []
    scores_deque = deque(maxlen=experiment_config["log_interval"])
    loss_deque = deque(maxlen=experiment_config["log_interval"])

    if logger is not None:
        logger.log_parameters(experiment_config)

    for episode in range(1, total_episodes + 1):
        rewards = []
        state = env.reset()

        for t in range(experiment_config["max_steps_per_episode"]):
            action = agent.act(state.unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            #
            # print(
            #     f"Timestep = {t} \n\tState: {state} \n\tReward: {reward} \n\tAction: {action}"
            # )

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


def eval_agent(agent, env, total_episodes=10000, logger=None):
    scores = []
    scores_deque = deque(maxlen=experiment_config["log_interval"])

    if logger is not None:
        logger.log_parameters(experiment_config)

    for episode in range(1, total_episodes + 1):
        rewards = []
        state = env.reset()

        for t in range(experiment_config["max_steps_per_episode"]):
            action = agent.act(state.unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            state = next_state

            if done:
                break

        agent.reset()

        episode_length = len(rewards)
        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))

        if episode % experiment_config["log_interval"] == 0:
            metrics = {
                "(Eval) Average Score": np.mean(scores_deque),
            }
            if logger:
                log_to_comet_ml(logger, metrics, step=episode)
            metrics.update({"Episode": episode})
            log_to_screen(metrics)
