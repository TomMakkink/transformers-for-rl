ppo_config = {
    "steps_per_epoch":2048, 
    "batch_size": 20, 
    "lr":1e-3, 
    "gamma":0.99, 
    "clip_ratio":0.2, 
    "train_iters":4, 
    "lam":0.97, 
    "ent_coef":0.00, 
    "value_coef":0.5, 
} 