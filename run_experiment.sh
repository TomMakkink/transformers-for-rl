#!/bin/bash
 
# simple example for running a single experiment
python experiment.py --agent a2c --env memory_size/5 --num_eps 1000 --seed 42 --memory lstm
# python graphs.py
