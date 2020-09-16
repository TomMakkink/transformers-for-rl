dqn_config = {
    "lr": 1e-3,
    "gamma": 0.99,
    "buffer_size": 1000,
    "batch_size": 32,
    "epsilon": {"start": 1.0, "final": 0.01, "decay": 500},
    "warm_up_timesteps": 1000,
    "target_update": 100,
}
