dqn_config = {
    "lr": 3e-4,
    "gamma": 0.99,
    "buffer_size": 1000,
    "batch_size": 64,
    "epsilon": {"start": 1.0, "final": 0.01, "decay": 12800},
    "warm_up_timesteps": 4096,
    "target_update": 5,
}
