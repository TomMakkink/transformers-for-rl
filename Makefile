USE_GPU=False

GPUS=--gpus all
SHM=--shm-size=8gb
PORT=8889
TENSORBOARD_PORT=6003
PORTS=-p $(PORT):$(PORT) -p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT)

CONTAINER=transformers
BASE_FLAGS=-it --rm --init $(PORTS) -v $(PWD):/wd -w /wd --name $(CONTAINER)

ifeq ($(USE_GPU), True)
	RUN_FLAGS=$(GPUS) $(SHM) $(BASE_FLAGS)
else
	RUN_FLAGS=$(BASE_FLAGS)
endif

DOCKER_RUN=docker run $(RUN_FLAGS) $(CONTAINER)
DOCKER_BUILD=cd Docker && docker build -t $(CONTAINER) -f Docker.$(CONTAINER) .
DEFAULT=bash
EXP=bash run_experiment.sh
PLOT=python graphs.py

up:
	$(DOCKER_RUN) $(DEFAULT)

run:
	$(DOCKER_RUN) $(EXP)
	
plot:
	$(DOCKER_RUN) $(PLOT)

build:
	$(DOCKER_BUILD)