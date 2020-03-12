docker run --gpus all --shm-size 8G -p 8887:8887 -p 6008:6008 -it -v "$(pwd)":/wd transformer:latest bash
