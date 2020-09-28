from utils.logging import set_up_comet_ml, log_to_screen
from utils.utils import (
    update_configs,
    get_agent,
    set_random_seed,
    set_device,
    create_environment,
    get_sweep_from_bsuite_id,
    get_save_path,
)
from utils.visualisation import (viz_forget_activation, viz_attention, plot_lstm_gates,
                                 plot_lstm_forget_activation_heat_map)

import argparse
from experiments.agent_trainer import train_agent, eval_agent
import torch


def run(args):
    rl_agent = get_agent(args.agent)
    set_device()
    set_random_seed(args.seed)

    env_id_list = get_sweep_from_bsuite_id(args.env)
    for env_id in env_id_list:
        if args.comet:
            tags = [
                args.agent,
                args.memory,
                f"seed={args.seed}",
                env_id,
                f"window={args.window}",
            ]
            logger = set_up_comet_ml(tags=[*tags])
        else:
            logger = None
        env = create_environment(
            agent=args.agent,
            seed=args.seed,
            memory=args.memory,
            env=env_id,
            window_size=args.window,
        )
        action_size = env.action_space.n
        state_size = env.observation_space.shape[1]
        agent = rl_agent(state_size, action_size, memory=args.memory)
        total_episodes = (
            env.bsuite_num_episodes if args.num_eps is None else args.num_eps
        )
        train_agent(agent, env, total_episodes, logger)
        save_path = get_save_path(args.window, args.agent, args.memory)
        file_name = env_id.replace("/", "_") + "_saved_model.pt"
        torch.save(agent, save_path + file_name)


def eval(args):
    print("Starting Eval run ...")
    save_path = get_save_path(args.window, args.agent, args.memory)

    env_ids = get_sweep_from_bsuite_id(args.env)

    for env_id in env_ids:
        env = create_environment(
            agent=args.agent,
            seed=args.seed,
            memory=args.memory,
            env=env_id,
            window_size=args.window,
            eval_run=args.eval
        )
        env_id = env_id.replace("/", "_")

        file_name = env_id + "_saved_model.pt"

        agent = torch.load(save_path + file_name)
        total_episodes = (
            env.bsuite_num_episodes if args.num_eps is None else args.num_eps
        )
        eval_agent(agent, env, total_episodes)


def plot_viz(args):
    save_path = get_save_path(args.window, args.agent, args.memory)

    env_ids = get_sweep_from_bsuite_id(args.env)

    for id in env_ids:
        env_id = id.replace("/", "_")

        file_name = env_id + "_saved_model.pt"

        agent = torch.load(save_path + file_name)

        viz_data = agent.net.memory_network.visualisation_data[:-1]

        if args.memory is not None:
            if args.memory == 'lstm':
                # viz_forget_activation(viz_data, env_id, args.agent, args.window)
                plot_lstm_gates(viz_data, env_id, args.agent, args.window)
                plot_lstm_forget_activation_heat_map(viz_data, env_id, args.agent,
                                                     args.window)
            else:

                viz_attention(viz_data, env_id, args.agent, args.window, args.memory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="transformers-for-rl")
    parser.add_argument("--name", type=str, default="Test")
    parser.add_argument("--agent", type=str)
    parser.add_argument("--memory", type=str, default=None)
    parser.add_argument("--num_eps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--tags", nargs="*", help="Additional comet experiment tags.")
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_mlp", type=int, default=32)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    update_configs(args)

    if args.eval:
        eval(args)

    else:
        run(args)

        if args.viz:
            print("Plotting viz")
            plot_viz(args)


if __name__ == "__main__":
    main()
