import hydra
from omegaconf import DictConfig
from utils import (
    plot_bsuite_graph,
    plot_training_results,
    plot_evaluation_results,
    plot_attention_weights,
)


@hydra.main(config_path="configs/", config_name="graphs")
def main(cfg: DictConfig):
    # if cfg.plot_bsuite:
    #     print("Plotting bsuite...")
    #     plot_bsuite_graph(cfg.memory_models, cfg.envs, cfg.seeds, "training")
    if cfg.plot_training:
        print("Plotting training...")
        plot_training_results(cfg.memory_models, cfg.envs, seeds=cfg.seeds)
    if cfg.plot_eval:
        print("Plotting evaluation...")
        plot_evaluation_results(cfg.memory_models, cfg.envs, seeds=cfg.seeds)
    # if cfg.plot_attn_weights:
    #     print("Plotting attention weights...")
    #     plot_attention_weights(
    #         cfg.memory_models,
    #         cfg.envs,
    #         seeds=cfg.seeds,
    #         plot_frequency=cfg.plot_frequency,
    #     )


if __name__ == "__main__":
    main()
