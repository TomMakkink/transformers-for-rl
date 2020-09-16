#!/bin/bash
 
# simple example for running a single experiment
python experiment.py --agent a2c --env memory_len/4 --memory mha --window 6 --num_eps 100 --viz --seed 0
# python graphs.py
