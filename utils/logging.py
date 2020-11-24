import comet_ml


def set_up_comet_ml(project_name, experiment_name, tags: list):
    logger = comet_ml.Experiment(
        project_name=project_name,
        log_code=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_host=False,
    )
    logger.set_name(experiment_name)
    for tag in tags:
        logger.add_tag(tag)

    return logger


def log_to_comet_ml(experiment, metrics, step):
    for name, value in metrics.items():
        experiment.log_metric(name, value, step=step)


def log_to_tensorboard(writer, metrics, step):
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)


def log_to_screen(metrics):
    print("------------------------------")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("------------------------------")


def log_episode_returns(folder, env_id, data):
    filename = folder + f"{env_id}_rewards.pt"
    fileObject = open(filename, "wb")
    pickle.dump(data, fileObject)
    fileObject.close()
