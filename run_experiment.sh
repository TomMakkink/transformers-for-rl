#!/bin/bash

# simple example for running a single experiment
python experiment.py --agent a2c --env memory_length --window 100 --memory mha --comet --num_eps 1000
# python graphs.py 
