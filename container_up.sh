docker run --gpus all --shm-size 8G -p 8888:8888 -p 6006:6006 -it -v "$(pwd)":/wd transformer:latest bash
