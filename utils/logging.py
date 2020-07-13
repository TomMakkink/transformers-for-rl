import comet_ml
from configs.experiment_config import experiment_config

def set_up_comet_ml(tags:list):
    logger = comet_ml.Experiment(project_name=experiment_config['project_name'], log_code=False,
                                     log_git_metadata=False, log_git_patch=False, log_env_host=False)
    logger.set_name(experiment_config['experiment_name'])
    for tag in tags: 
        logger.add_tag(tag)
        
    return logger 


def log_to_comet_ml(experiment, metrics, step):
    for name, value in metrics.items(): 
        experiment.log_metric(name, value, step=step)
    
    # experiment.log_metric('Mean Episode Reward', mean_episode_returns, step=total_time_steps)
    # experiment.log_metric('Mean Episode Length', mean_episode_length, step=total_time_steps)
    # experiment.log_metric('Actor Loss', loss_actor.item(), step=total_time_steps)
    # experiment.log_metric('Critic Loss', loss_critic.item(), step=total_time_steps)
    # experiment.log_metric('Loss', loss.item(), step=total_time_steps)
    # experiment.log_metric('Entropy', entropy, step=total_time_steps)
    # experiment.log_metric('Kl', kl, step=total_time_steps)


def log_to_tensorboard(writer, metrics, step):
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)
    # writer.add_scalar('Mean Episode Reward',
    #                         mean_episode_returns, total_time_steps)
    # writer.add_scalar('Mean Episode Length',
    #                         mean_episode_length, total_time_steps)
    # writer.add_scalar('Loss/Loss', loss, total_time_steps)
    # writer.add_scalar(
    #     'Loss/Actor Loss', loss_actor, total_time_steps)
    # writer.add_scalar(
    #     'Loss/Critic Loss', loss_critic, total_time_steps)
    # writer.add_scalar('Extra/Kl', kl, total_time_steps)
    # writer.add_scalar('Extra/Entropy', ent, total_time_steps)


def log_to_screen(metrics):
    print("------------------------------")
    for name, value in metrics.items(): 
        print(f"{name}: {value:.2f}")
    # print(f"Mean episode returns: {mean_episode_returns:.2f}")
    # print(f"Mean episode length: {mean_episode_length}")
    # print(f"Loss actor: {loss_actor:.2f}")
    # print(f"Loss critic: {loss_critic:.2f}")
    # print(f"Loss: {loss:.2f}")
    # print(f"Ent: {ent:.2f}")
    # print(f"KL: {kl:.2f}")
    # print(f"Timestep: {total_time_steps}")
    # print(f"Episodes: {total_episodes}")
    print("------------------------------")