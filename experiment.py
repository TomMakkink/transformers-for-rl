from experiments.cartpole import cartpole_test
# from experiments.ppo_test import ppo_test

if __name__ == '__main__':
    cartpole_test("CartPole/PPO_Baseline", total_timesteps=50000, seed=10)
