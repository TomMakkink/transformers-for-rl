import hydra
from omegaconf import DictConfig
from utils import plot_bsuite_graph, plot_training_results, plot_evaluation_results


@hydra.main(config_path="configs/", config_name="graphs")
def main(cfg: DictConfig):
    print("Plotting results...")
    if cfg.plot_bsuite:
        plot_bsuite_graph(cfg.memory_models, cfg.envs, cfg.window, "training")
    if cfg.plot_training:
        plot_training_results(cfg.memory_models, cfg.envs, window=cfg.window)
    if cfg.plot_eval:
        plot_evaluation_results(cfg.memory_models, cfg.envs, window=cfg.window)


if __name__ == "__main__":
    main()
