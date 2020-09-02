# transformers-for-rl
Investigating the use of various Transformer architectures in Reinforcement Learning. 

# Running Experiments: 
```console
foo@bar:~$ python experiment.py --agent a2c --memory gtrxl --env memory_len/0 --window 10
```

# Docker: 
Build the Docker file: 
```console
foo@bar:~$ cd Docker 
foo@bar:~$ docker build -t transformers -f Docker.transformer .
```

Run the Docker image:

Without GPU:  
```console
foo@bar:~$ docker run -p 8889:8889 -p 6003:6003 -it -v "$(pwd)":/wd/ transformers:latest bash
```
With GPU: 
```console
foo@bar:~$ docker run --gpus all --shm-size 8G -p 8889:8889 -p 6003:6003 -it -v "$(pwd)":/wd/ transformers:latest bash
```


