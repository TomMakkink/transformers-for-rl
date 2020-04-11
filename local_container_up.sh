docker run --gpus all --shm-size 8G -p 8887:8887 -p 6007:6007 -it -v "$(pwd)":/wd/ transformers:latest bash


