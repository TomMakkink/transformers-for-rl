import torch 

transformer_config = {
    "d_model": 3, 
    "output_dim": 512, 
    "num_heads": 1, 
    "num_layers": 8, 
    "dim_mlp": 128, 
    "dropout": 0.1, 
    "mem_len": 0.0, 
    "transformer_type": "reformer"}