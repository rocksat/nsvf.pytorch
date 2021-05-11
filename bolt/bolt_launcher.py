import argparse
import os

import turibolt as bolt


def generate_bolt_config():
    """
    generate common configuration for all tasks
    """
    bolt_config = dict()
    bolt_config['name'] = 'nsvf.pytorch'
    bolt_config['description'] = 'Neural Sparse Voxel Fields (NSVF)'
    bolt_config['command'] = 'python3 spawn_child_tasks.py'
    bolt_config['setup_command'] = 'bash setup.sh'
    bolt_config['priority'] = 1
    bolt_config['is_parent'] = True
    bolt_config['tags'] = ['NSVF']

    # permission
    bolt_config['permissions'] = {
        'owners': 'chuang5',
        'viewers': 'vejasper-dev'
    }

    # resources
    bolt_config['resources'] = {
        'num_gpus': 1,
        'group': 'p6:hw:ve',
        'cluster': 'simcloud-mr2.apple.com',
        'timeout': '14d',
        'ports': ["TENSORBOARD_PORT"]
    }

    # cluster options
    bolt_config['cluster_options'] = {'simcloud': {'smi': 'current-ubuntu18.04-cuda11.0'}}

    return bolt_config


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Neural Sparse Voxel Fields")
    parser.add_argument("--video_list",
                        required=True,
                        type=str,
                        help='udf bundles to train')
    return parser.parse_args(argv)


def main(args):
    bolt_config = generate_bolt_config()

    # load video list
    assert os.path.exists(args.video_list)
    with open(args.video_list, 'r') as f:
        all_videos = f.read().splitlines()

    bolt_config['parameters'] = {'videos_list': all_videos}
    bolt.submit(bolt_config, tar='.', max_retries=3)


if __name__ == '__main__':
    args = get_args()
    main(args)
