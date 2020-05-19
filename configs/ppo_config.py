ppo_config = {
    "steps_per_epoch":2000, 
    "batch_size": 2000, 
    "lr":0.0003, 
    "gamma":0.99, 
    "clip_ratio":0.2, 
    "train_iters":80, 
    "lam":0.95, 
    "ent_coef":0.00, 
    "value_coef":0.5, 
    "save_freq":10,
    # "image_pad": 4, 
} 