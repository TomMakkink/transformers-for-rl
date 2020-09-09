#!/bin/bash
# Type Checking 
# pytype experiment.py
# env='memory_size'
# declare -a envs=("memory_size" "memory_len")
 
# simple example for running a single experiment
python experiment.py --agent a2c --memory gtrxl --env memory_len/0 --window 10

# Running multiple experiments
# Read the array values with space
# for env in "${envs[@]}"; do
#   for i in {0..5}; do 
#     python experiment.py --env "${env}/${i}" --transformer vanilla --window 20 --name Canonical --comet 
#     python experiment.py --env "${env}/${i}" --transformer gtrxl --window 20 --name GTrXL --comet 
#     # python experiment.py --env "${env}/${i}" --transformer xl --window 4 --name XL --comet 
#     # python experiment.py --env "${env}/${i}" --transformer rezero --window 4 --name ReZero --comet
#     # python experiment.py --env "${env}/${i}" --name LSTM --lstm --comet 
#     # python experiment.py --env "${env}/${i}" --name Actor_Critic --comet
#     python experiment.py --env "${env}/${i}" --transformer linformer --window 20 --name Linformer --comet 
#   done
# done

# Seed = 1
# python experiment.py --name Canonical --transformer vanilla --seed 1 & 
# python experiment.py --name ReZero --transformer rezero --seed 1 & 
# python experiment.py --name GTrXL --transformer gtrxl --seed 1 
# python experiment.py --name XL --transformer xl --seed 1 

# # Seed = 2
# python experiment.py --name Canonical --transformer vanilla --seed 2 & 
# python experiment.py --name ReZero --transformer rezero --seed 2 & 
# python experiment.py --name GTrXL --transformer gtrxl --seed 2 
# python experiment.py --name XL --transformer xl --seed 2 

# # Seed = 3
# python experiment.py --name Canonical --transformer vanilla --seed 3 & 
# python experiment.py --name ReZero --transformer rezero --seed 3 & 
# python experiment.py --name GTrXL --transformer gtrxl --seed 3 
# python experiment.py --name XL --transformer xl --seed 