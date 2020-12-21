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
    logging_cfg: DictConfig,
):
    use_cuda = experiment_info.device == "cuda"
    set_random_seed(experiment_info.seed, use_cuda)
    log_to_screen(experiment_info)

    env_id_list = get_sweep_from_bsuite_id(env_cfg.name)

    for env_id in env_id_list:
        if logging_cfg.use_comet:
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
            save_dir="data/training/",
        )

        agent = build_agent(
            state_size=env.observation_space.shape[1],
            action_size=env.action_space.n,
            agent_cfg=agent_cfg,
            memory_cfg=memory_cfg,
            device=experiment_info.device,
        )

        training_metric_save_path = f"data/training/{env_id.replace('/', '-')}_log.csv"
        agent = train_agent(
            agent=agent,
            env=env,
            total_episodes=experiment_info.total_episodes,
            max_steps_per_episode=env_cfg.max_steps_per_episode,
            logger=logger,
            log_interval=logging_cfg.log_interval,
            save_path=training_metric_save_path,
        )

        if logging_cfg.plot_viz:
            plot_viz(
                save_dir="plots/training/",
                memory_name=memory_cfg.name,
                env=env_id,
                agent=agent,
                plot_frequency=logging_cfg.plot_frequency,
            )

        if experiment_info.eval_agent:
            env = build_env(
                env=env_id,
                window=env_cfg.window,
                device=experiment_info.device,
                save_dir="data/eval/",
            )

            eval_metric_save_path = f"data/eval/{env_id.replace('/', '-')}_log.csv"
            evaluate_agent(
                agent=agent,
                env=env,
                total_episodes=experiment_info.eval_episodes,
                save_path=eval_metric_save_path,
            )

            if logging_cfg.plot_viz:
                plot_viz(
                    save_dir="plots/eval/",
                    memory_name=memory_cfg.name,
                    env=env_id,
                    agent=agent,
                    plot_frequency=logging_cfg.plot_frequency,
                )


def train_agent(
    agent,
    env,
    save_path,
    total_episodes=10000,
    max_steps_per_episode=1000,
    logger=None,
    log_interval=10,
):
    scores = []
    save_metrics = []
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
            ave_score = np.mean(scores_deque)
            ave_loss = np.mean(loss_deque)
            metrics = {
                "Average Score": ave_score,
                "Loss": ave_loss,
            }
            # if logger:
            #     log_to_comet_ml(logger, metrics, step=episode)
            metrics.update({"Episode": episode})
            log_to_screen(metrics)
            save_metrics.append(np.array([episode, ave_score, ave_loss]))

    save_metrics = np.array(save_metrics)
    np.savetxt(
        save_path,
        save_metrics,
        delimiter=",",
        header="Episode, Average Score, Loss",
        comments="",
        fmt="%f",
    )

    return agent


def evaluate_agent(
    agent,
    env,
    save_path,
    total_episodes=100,
    max_steps_per_episode=1000,
    logger=None,
    log_interval=100,
):
    print("Evaluating agent...")
    scores = []
    scores_deque = deque(maxlen=log_interval)
    save_metrics = []

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

        agent.reset()

        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))

        if episode % log_interval == 0:
            ave_score = np.mean(scores_deque)
            metrics = {
                "Average Score": ave_score,
            }
            if logger:
                log_to_comet_ml(logger, metrics, step=episode)
            metrics.update({"Episode": episode})
            log_to_screen(metrics)
            save_metrics.append(np.array([episode, ave_score]))

    save_metrics = np.array(save_metrics)
    np.savetxt(
        save_path,
        save_metrics,
        delimiter=",",
        header="Episode, Average Score",
        comments="",
        fmt="%f",
    )
