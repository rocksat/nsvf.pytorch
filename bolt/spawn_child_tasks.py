import turibolt as bolt


def main():
    config = bolt.get_current_config()
    all_videos = config['parameters'].pop('videos_list')

    # update config file
    config['is_parent'] = False
    config['command'] = 'bash run.sh'
    config['resources']['num_gpus'] = 4

    # environment variables
    config['environment_variables'] = {
        'PYTHONPATH': '/task_runtime/ostools:$PYTHONPATH',
    }

    for video in all_videos:
        config['environment_variables']['DATA'] = video
        bolt.submit(config, tar='.', max_retries=3)


if __name__ == '__main__':
    main()
