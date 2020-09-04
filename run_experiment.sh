#!/bin/bash
# Type Checking 
# pytype experiment.py
env='memory_size'
for i in {0..10}
  do 
    python experiment.py --env "${env}/${i}" --transformer vanilla --window 4 --name Canonical --comet 
    python experiment.py --env "${env}/${i}" --transformer gtrxl --window 4 --name GTrXL --comet 
    python experiment.py --env "${env}/${i}" --transformer xl --window 4 --name XL --comet 
    python experiment.py --env "${env}/${i}" --transformer rezero --window 4 --name ReZero --comet
    python experiment.py --env "${env}/${i}" --name LSTM --lstm --comet 
    python experiment.py --env "${env}/${i}" --name Actor_Critic --comet
 done




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