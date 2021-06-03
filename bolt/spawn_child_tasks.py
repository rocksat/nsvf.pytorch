import turibolt as bolt


def main():
    config = bolt.get_current_config()
    all_videos = config['parameters'].pop('videos_list')

    # update config file
    config['is_parent'] = False
    config['command'] = 'bash run.sh'
    config['resources']['cluster'] = 'apc_usuqo27'
    config['resources']['group'] = 'hwt:is:ve:rtcv'
    config['resources']['docker_image'] = 'docker.apple.com/ad-algo/base-devel-tensorflow-optimized:0.1.1'
    config['resources']['disk_gb'] = 800.0
    config['resources']['memory_gb'] = 760.0
    config['resources']['num_cpus'] = 80
    config['resources']['num_gpus'] = 8
    config.pop('cluster_options')
    # environment variables
    config['environment_variables'] = {
        'PYTHONPATH': '/task_runtime/ostools:$PYTHONPATH',
        'PATH': '/usr/local/cuda-10.1/bin:$PATH',
    }

    for video in all_videos:
        config['environment_variables']['DATA'] = video
        bolt.submit(config, tar='.', max_retries=3)


if __name__ == '__main__':
    main()
