import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--disable', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default='/data/xpf/datasets')
    parser.add_argument('--sample_path', type=str, default='./samples')

    parser.add_argument('--data_name', type=str, default='imagenet10', choices=['cifar10', 'imagenet10'])
    parser.add_argument('--model_name', type=str, default='vgg16')

    parser.add_argument('--attack_name', type=str, default='blended')
    parser.add_argument('--trigger', type=str, default='0')
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.02)

    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--samples_idx', type=str, default='cifar10_vgg16_blended_0_0_0.02_10_0.5')

    opts = parser.parse_args()
    return opts
