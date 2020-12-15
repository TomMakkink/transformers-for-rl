from collections import deque
import numpy as np
from omegaconf import DictConfig
from utils import (
    log_to_screen,
    log_to_comet_ml,
    set_up_comet_ml,
    set_random_seed,
    get_sweep_from_bsuite_id,
    build_agent,
    build_env,
    plot_viz,
)


def run(
    experiment_info: DictConfig,
    env_cfg: DictConfig,
    agent_cfg: DictConfig,
    memory_cfg: DictConfig,
):
    use_cuda = experiment_info.device == "cuda"
    set_random_seed(experiment_info.seed, use_cuda)

    env_id_list = get_sweep_from_bsuite_id(env_cfg.name)

    for env_id in env_id_list:
        if experiment_info.use_comet:
            tags = [
                agent_cfg.name,
                memory_cfg.name,
                experiment_info.seed,
                env_id,
                f"seed={experiment_info.seed}",
                f"window={experiment_info.window}",
            ]
            logger = set_up_comet_ml(
                experiment_info.project_name,
                experiment_info.experiment_name,
                tags=tags,
            )
        else:
            logger = None

        env = build_env(
            env=env_id,
            window=env_cfg.window,
            device=experiment_info.device,
            save_dir="data/",
        )

        agent = build_agent(
            state_size=env.observation_space.shape[1],
            action_size=env.action_space.n,
            agent_cfg=agent_cfg,
            memory_cfg=memory_cfg,
            device=experiment_info.device,
        )

        log_to_screen(experiment_info)
        train_agent(
            agent=agent,
            env=env,
            total_episodes=experiment_info.total_episodes,
            max_steps_per_episode=env_cfg.max_steps_per_episode,
            logger=logger,
            log_interval=experiment_info.log_interval,
        )

        if experiment_info.plot_viz:
            plot_viz(
                save_dir="plots/",
                memory_name=memory_cfg.name,
                env=env_id,
                agent=agent,
                plot_frequency=experiment_info.plot_frequency,
            )


def train_agent(
    agent,
    env,
    total_episodes=10000,
    max_steps_per_episode=1000,
    logger=None,
    log_interval=10,
):
    scores = []
    scores_deque = deque(maxlen=log_interval)
    loss_deque = deque(maxlen=log_interval)

    for episode in range(1, total_episodes + 1):
        rewards = []
        state = env.reset()

        for t in range(max_steps_per_episode):
            action = agent.act(state.unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            agent.collect_experience(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        loss = agent.optimize_network()
        agent.reset()

        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))
        loss_deque.append(loss)

        if episode % log_interval == 0:
            metrics = {
                "Average Score": np.mean(scores_deque),
                "Loss": np.mean(loss_deque),
            }
            if logger:
                log_to_comet_ml(logger, metrics, step=episode)
            metrics.update({"Episode": episode})
            log_to_screen(metrics)
