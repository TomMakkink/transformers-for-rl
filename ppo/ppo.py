import numpy as np
import torch
from torch.optim import Adam
import gym
from gym.wrappers import Monitor
import time
import datetime
from utils import PPOBuffer, MLPActorCritic, count_vars, combined_shape, discount_cumsum, plot_grad_flow
from torch.utils.tensorboard import SummaryWriter

# GPU or CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Summary Writer 
writer = SummaryWriter("runs/" + str(datetime.datetime.now()))

def ppo(env_name, actor_critic=MLPActorCritic, seed=3, 
        steps_per_epoch=1000, epochs=2000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 
    with early stopping based on approximate KL
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """
    # Random seed
    # seed += 10000 * proc_id()
    env = gym.make(env_name)
    env = Monitor(env, './video', force=True)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space).to(device)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch)
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        # Change obs from [batch_size, Width, Height, Channel] -> [batch_size, Features]
        state = torch.flatten(obs, start_dim=1).to(device)
        pi, logp = ac.pi(state, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        # Change obs from [batch_size, Width, Height, Channel] -> [batch_size, Features]
        state = torch.flatten(obs, start_dim=1)
        return ((ac.v(state) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)


    def update(epoch):
        data = buf.get(device)

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            # plot_grad_flow(ac.pi.named_parameters())
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # plot_grad_flow(ac.v.named_parameters())
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        writer.add_scalar("Actor (Policy) loss", pi_l_old, epoch)
        writer.add_scalar("Critic (Value) loss", v_l_old, epoch)
        writer.add_scalar("KL", kl, epoch)
        writer.add_scalar("Entropy", ent, epoch)
        writer.add_scalar("Delta Actor (Policy) loss", (loss_pi.item() - pi_l_old), epoch)
        writer.add_scalar("Delta Critic (Value) loss", (loss_v.item() - v_l_old), epoch)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        episode_rewards = []
        episode_lengths = []
        for t in range(local_steps_per_epoch):
            # Convert to tensor and add single dimension to the beginning 
            state = torch.from_numpy(o.copy()).float().unsqueeze(0).to(device)
            state = torch.flatten(state, start_dim=1)
            a, v, logp = ac.step(state)
            next_o, r, d, _ = env.step(a.squeeze())
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    state = torch.from_numpy(o.copy()).float().unsqueeze(0).to(device)
                    state = torch.flatten(state, start_dim=1)
                    _, v, _ = ac.step(state)
                else:
                    v = 0
                buf.finish_path(v)
                episode_rewards.append(ep_ret)
                episode_lengths.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Perform PPO update!
        update(epoch)

        # Track mean episode return per epoch 
        mean_episode_reward = sum(episode_rewards)/len(episode_rewards)
        mean_episode_length = sum(episode_lengths)/len(episode_lengths)
        writer.add_scalar('Mean Episode Reward', mean_episode_reward, epoch)
        writer.add_scalar('Mean Episode Length', mean_episode_length, epoch)
    
    writer.close()

if __name__ == '__main__':
    # mpi_fork(args.cpu)  # run parallel code with mpi

    ppo('CarRacing-v0', actor_critic=MLPActorCritic, seed=1)
