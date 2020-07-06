import comet_ml

from experiments.cartpole import cartpole_test

# from experiments.ppo_test import ppo_test

if __name__ == '__main__':
    name = "CartPole/PPO_Baseline"
    experiment = comet_ml.Experiment(project_name="transformers-for-rl", log_code=False,
                                        log_git_metadata=False, log_git_patch=False, log_env_host=False)
    experiment.add_tag(name)
    cartpole_test(name, experiment, total_timesteps=50000, seed=10)
