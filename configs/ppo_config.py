ppo_config = {
    "steps_per_epoch":500, 
    "epochs":1000, 
    "gamma":0.99, 
    "clip_ratio":0.2, 
    "pi_lr":3e-4,
    "vf_lr":1e-3, 
    "train_pi_iters":10, 
    "train_v_iters":10, 
    "lam":0.97, 
    "max_ep_len":1000,
    "target_kl":0.01, 
    "save_freq":10,
    "num_stack":4, 
    "repeat_action": 4, 
} 
        