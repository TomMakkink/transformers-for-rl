#!/bin/bash

# simple example for running a single experiment
python experiment.py --agent a2c --env memory_custom --window 5 --num_eps 1 --memory vanilla
# python graphs.py 