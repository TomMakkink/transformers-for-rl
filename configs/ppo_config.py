ppo_config = {
    "steps_per_epoch":2000, 
    "epochs":5000, 
    "gamma":0.99, 
    "clip_ratio":0.2, 
    "lr":0.001, 
    "train_ppo_iters":10, 
    "lam":0.97, 
    "max_ep_len":1000,
    "ent_coef":0.01, 
    "value_coef":0.5, 
    "save_freq":10,
    "num_stack":4, 
    "repeat_action": 4, 
} 
        