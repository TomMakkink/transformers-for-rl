dqn_config = {
    "max_steps_per_episode": 2048,
    "lr": 0.0001,
    "gamma": 0.99,
    "log_interval": 10,
    "buffer_size": 10000,
    "batch_size": 32,
    "epsilon": {"start": 1.0, "final": 0.01, "decay": 500},
}
