from utils.logging import set_up_comet_ml
from utils.utils import (
    set_random_seed,
    create_environment,
    get_sweep_from_bsuite_id,
    get_save_path,
)
from utils.visualisation import viz_forget_activation, viz_attention, plot_lstm_gates
from omegaconf import DictConfig, OmegaConf
import hydra

import argparse
import torch

# @hydra.main(config_path='configs/', config_name='experiment')
# def run(cfg: DictConfig) -> None:
#     set_random_seed(cfg.seed)
#
#     env_id_list = get_sweep_from_bsuite_id(cfg.env)
#     for env_id in env_id_list:
#         if cfg.comet:
#             tags = [
#                 cfg.agent.name,
#                 cfg.memory.name,
#                 f"seed={cfg.seed}",
#                 env_id,
#                 f"window={cfg.window}",
#             ]
#             logger = set_up_comet_ml(cfg.project_name, cfg.experiment_name, tags=[*tags])
#         else:
#             logger = None
#
#         env = create_environment(
#             agent=cfg.agent.name,
#             seed=cfg.seed,
#             memory=cfg.memory.name,
#             env=cfg.env,
#             window_size=cfg.window,
#             device=cfg.device,
#         )
#
#         if cfg.memory._target_ is not None:
#             memory = hydra.utils.instantiate(cfg.memory, input_dim=cfg.agent.hidden_size[-1], output_dim=cfg.agent.hidden_size[-1])
#         else:
#             memory = None
#
#         action_size = env.action_space.n
#         state_size = env.observation_space.shape[1]
#         agent = hydra.utils.instantiate(cfg.agent, state_size=state_size, action_size=action_size, memory=memory, device=cfg.device)
#         total_episodes = (
#             env.bsuite_num_episodes if cfg.num_eps is None else cfg.num_eps
#         )
#
#         train_agent(agent, env, total_episodes, cfg.max_steps_per_episode, logger=None)

#     save_path = get_save_path(args.window, args.agent, args.memory)
#     file_name = env_id.replace("/", "_") + "_saved_model.pt"
#     torch.save(agent, save_path + file_name)


#
#
# def plot_viz(args):
#     save_path = get_save_path(args.window, args.agent, args.memory)
#
#     env_ids = get_sweep_from_bsuite_id(args.env)
#
#     for id in env_ids:
#         env_id = id.replace("/", "_")
#
#         file_name = env_id + "_saved_model.pt"
#
#         agent = torch.load(save_path + file_name)
#
#         viz_data = agent.net.memory_network.visualisation_data[:-1]
#
#         if args.memory is not None:
#             if args.memory == 'lstm':
#                 # viz_forget_activation(viz_data, env_id, args.agent, args.window)
#                 plot_lstm_gates(viz_data, env_id, args.agent, args.window)
#             else:
#                 viz_attention(viz_data, env_id, args.agent, args.window, args.memory)


from trainers.agent_trainer import run


@hydra.main(config_path="configs/", config_name="experiment")
def main(cfg: DictConfig) -> None:
    run(cfg.experiment_info, cfg.env, cfg.agent, cfg.memory)

    # set_up_experiment(cfg.experiment_info)
    # env_id_list = get_sweep_from_bsuite_id(cfg.env.name)
    #
    # env = build_env(env_cfg=cfg.env)
    #
    # agent = build_agent(
    #     state_size=env.observation_space.shape[1],
    #     action_size=env.action_space.n,
    #     agent_cfg=cfg.agent,
    #     memory_cfg=cfg.memory,
    # )
    # # agent.train()

    # run()

    # if args.viz:
    #     print("Plotting viz")
    #     plot_viz(args)


if __name__ == "__main__":
    main()
