"""Inspect the statistics of AMC528"""
import numpy as np
from argparse import ArgumentParser
from T2_ADC.utils.my_func import norm_01, norm2, norm1
import pylab as plt


def get_args():
    """Get arguments"""
    parser = ArgumentParser('Get args.')
    parser.add_argument('--dir_in', type=str, default=r'F:\BACKUPS\AMC_GAN_T2_ADC\inputs/')
    parser.add_argument('--im_name', type=str, default='data_528')
    parser.add_argument('--load_mask', type=bool, default=True)
    parser.add_argument('--plot_input_stats_norm1', type=bool, default=False)
    parser.add_argument('--plot_input_stats_norm2', type=bool, default=False)
    parser.add_argument('--norm1', type=bool, default=True)
    parser.add_argument('--norm2', type=bool, default=False)
    parser.add_argument('--print_stats', type=bool, default=False)
    parser.add_argument('--show_im', type=bool, default=True)
    parser.add_argument('--othr', type=float, default=1e-6)
    parser.add_argument('--num_im_shown', type=int, default=10)
    return parser.parse_args()


def plot_input_stats(stats, axis=0, norm_type='norm2'):
    """plot inputs distribution
    axis = 0 if T2, 1 if ADC
    stats: output from norm1"""
    # in 4 dim, 0: mean, 1: std
    # 3rd dimension: pre_grubbs, post_grubbs, post_norm
    plt.figure()
    num_rows = 3 if norm_type == 'norm1' else 2
    plt.subplot(num_rows, 2, 1), plt.bar(np.arange(stats.shape[0]), stats[:, axis, 0, 0])
    plt.subplot(num_rows, 2, 2), plt.bar(np.arange(stats.shape[0]), stats[:, axis, 0, 1])
    plt.subplot(num_rows, 2, 3), plt.bar(np.arange(stats.shape[0]), stats[:, axis, 1, 0])
    plt.subplot(num_rows, 2, 4), plt.bar(np.arange(stats.shape[0]), stats[:, axis, 1, 1])
    if norm_type == 'norm1':
        plt.subplot(325), plt.bar(np.arange(stats.shape[0]), stats[:, axis, 2, 0])
        plt.subplot(326), plt.bar(np.arange(stats.shape[0]), stats[:, axis, 2, 1])


if __name__ == "__main__":
    args = get_args()
    im = np.load('%s%s.npy' % (args.dir_in, args.im_name))
    if args.load_mask:
        mask = np.load('%smask_528.npy' % args.dir_in)
    print(im.shape, mask.shape)

    if args.plot_input_stats_norm1 and args.norm1:
        im_norm, stats = norm1(im, return_stats=True)
        plot_input_stats(stats, norm_type='norm1')
    elif args.norm1:
        im_norm = norm1(im, mask=mask, othr=args.othr)
        plt.show()

    if args.norm2 and args.plot_input_stats_norm2:
        if 'im_norm' in locals():
            im_norm12, stats = norm2(im_norm, return_stats=True)
            plot_input_stats(stats)
        im_norm02, stats = norm2(im, return_stats=True)
        plot_input_stats(stats)
    else:
        if 'im_norm' in locals():
            im_norm12 = norm2(im_norm)
        plt.show()

    if args.print_stats:
        for j in range(im_norm.shape[0]):
            print(im_norm[j, 0].mean(), im_norm[j, 0].std(), im_norm[j, 0].min(), im_norm[j, 0].max())

    if args.show_im:
        for j in range(args.num_im_shown):
            plt.imshow(im_norm[j, 0], cmap='gray')
            plt.show()



