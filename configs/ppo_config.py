ppo_config = {
    "steps_per_epoch":1000, 
    "epochs":1, 
    "gamma":0.99, 
    "clip_ratio":0.2, 
    "lr":1e-3, 
    "train_ppo_iters":10, 
    "lam":0.97, 
    "max_ep_len":1000,
    "ent_coef":0.01, 
    "value_coef":0.5, 
    "save_freq":10,
    "num_stack":4, 
    "repeat_action": 4, 
} 
        