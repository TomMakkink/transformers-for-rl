# docker run --gpus all --shm-size 8G -p 8889:8889 -p 6003:6003 -it -v "$(pwd)":/wd/ transformers:latest bash
docker run --gpus all --shm-size 8G -p 8889:8889 -p 6003:6003 -it -v "$(pwd)":/wd/ transformers:latest bash

