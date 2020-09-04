dqn_config = {
    "lr": 0.0001,
    "gamma": 0.99,
    "buffer_size": 200,
    "batch_size": 32,
    "epsilon": {"start": 1.0, "final": 0.01, "decay": 500},
}
