tensorboard --logdir=runs --port 6003 & 
xvfb-run -a -s "-screen 0 1400x900x24" -- python experiment.py 