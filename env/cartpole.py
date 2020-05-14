import gym 

def make_env(env_name='CartPole-v0', max_episode_steps=500, seed=0):
    env = gym.make(env_name)
    env._max_episode_steps = max_episode_steps
    env.seed(seed)
    return env