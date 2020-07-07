#!/bin/bash
# Seed = 10
python experiment.py --name Canonical & 
python experiment.py --name ReZero --transformer rezero & 
python experiment.py --name GTrXL --transformer gtrxl

# tensorboard --logdir=runs --port 6003 & python experiment.py
# xvfb-run -a -s "-screen 0 1400x900x24" -- python experiment.py 
