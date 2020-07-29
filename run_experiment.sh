#!/bin/bash
# Type Checking 
# pytype experiment.py

# Seed = 1
python experiment.py --name Canonical --transformer vanilla --seed 1 & 
python experiment.py --name ReZero --transformer rezero --seed 1 & 
python experiment.py --name GTrXL --transformer gtrxl --seed 1 
python experiment.py --name XL --transformer xl --seed 1 

# Seed = 2
python experiment.py --name Canonical --transformer vanilla --seed 2 & 
python experiment.py --name ReZero --transformer rezero --seed 2 & 
python experiment.py --name GTrXL --transformer gtrxl --seed 2 
python experiment.py --name XL --transformer xl --seed 2 

# Seed = 3
python experiment.py --name Canonical --transformer vanilla --seed 3 & 
python experiment.py --name ReZero --transformer rezero --seed 3 & 
python experiment.py --name GTrXL --transformer gtrxl --seed 3 
python experiment.py --name XL --transformer xl --seed 