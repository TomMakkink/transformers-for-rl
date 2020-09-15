# transformers-for-rl
Investigating the use of various Transformer architectures in Reinforcement Learning. 

# Docker: 
Build the Docker file: 
```console
foo@bar:~$ make build
```

Run the Docker image:

Without GPU:  
```console
foo@bar:~$ make up
```
With GPU: 
```console
foo@bar:~$ make up USE_GPU=True
```

# Running Experiments: 
Specify the experiment you would like to run by editing the `run_experiments.sh` file. Then run the experiment using the following make command:
```console
foo@bar:~$ make run
```

# Plotting
```console
foo@bar:~$ make plot


