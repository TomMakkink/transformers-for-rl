import hydra
from omegaconf import DictConfig
from trainers.agent_trainer import run


@hydra.main(config_path="configs/", config_name="experiment")
def main(cfg: DictConfig) -> None:
    run(cfg.experiment_info, cfg.env, cfg.agent, cfg.memory)


if __name__ == "__main__":
    main()
