# coding=utf-8
"""Segmenting Prostate Using the model pretrained on NIH dataset"""
import numpy as np
import pylab as plt
import argparse
import os
import time
import mxnet as mx
from mxnet import gluon, nd
from T2_ADC.utils.my_func import Softmax, post_proc, norm1_v0


def parse_args():
    """Parse arguments from cmd"""
    parser = argparse.ArgumentParser('AMC Prostate Segmentation Prediction')
    parser.add_argument('--epoch', type=int, default=555)
    parser.add_argument('--run_id', type=int, default=11030, help='should be in 7000, 8000, 9000 if NIH, 11030 if PROMISE12')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--dir_checkpoints', type=str, default=r'F:\BACKUPS\NIH\outputs/')
    parser.add_argument('--dir_checkpoints', type=str, default=r'F:\BACKUPS\PROMISE2012\outputs\vnet1/')
    parser.add_argument('--dir_in_test', type=str, default=r'F:\BACKUPS\NIH\inputs/')
    parser.add_argument('--dir_in', type=str, default=r'F:\Minh\projects\AMC_v1\data\size192/')
    parser.add_argument('--dir_out', type=str, default=r'F:\Minh\projects\AMC_v1\prostate_segmentation/')
    parser.add_argument('--test_model', type=bool, default=False)
    parser.add_argument('--plot_pred', type=bool, default=True)
    parser.add_argument('--save_pred', type=bool, default=False)
    parser.add_argument('--only_T2', type=bool, default=True)
    return parser.parse_args()


def load_net(opts):
    """Load network and parameters from checkpoints"""
    _sym, arg_params, aux_params = mx.model.load_checkpoint('%srun%d/vnet_3d' % (opts.dir_checkpoints, opts.run_id), opts.epoch)
    sym = _sym().get_internals()['1_unit1_lastConv_bn_output']
    net = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))

    # Set the params
    net_params = net.collect_params()
    for param in arg_params:
        if param in net_params:
            net_params[param]._load_init(arg_params[param], ctx=opts.ctx)
    for param in aux_params:
        if param in net_params:
            net_params[param]._load_init(aux_params[param], ctx=opts.ctx)
    return net


def test_model(opts, net):
    """test model loaded from checkpoints"""
    im = np.load('%ssplit%d/im.npy' % (opts.dir_in_test, opts.split))
    lab = np.load('%ssplit%d/lab.npy' % (opts.dir_in_test, opts.split))

    print(im.shape)
    for sub_idx in range(22, 23):
        out = nd.squeeze(nd.argmax(net(nd.array(im[sub_idx: sub_idx + 1], ctx=opts.ctx)), axis=1)).asnumpy()
        out = post_proc(out, area_thr_slice=200)
        plot_(out, im[sub_idx], lab=lab[sub_idx])


def plot_(out, im, lab=None, num_rows=4, invisible=False, linewidth=.5):
    for j in range(out.shape[0]):
        plt.subplot(num_rows, int(np.ceil(out.shape[0]/num_rows)), j + 1)
        plt.imshow(im[0, j], cmap='gray')
        plt.axis('off')
        if out[j].sum() > 0:
            plt.hold(True)
            plt.contour(out[j], colors='r', linewidths=linewidth)
            if lab is not None:
                plt.contour(lab[j], colors='y', linewidths=linewidth)
    if not invisible:
        plt.show()
    plt.hold(False)


def load_dat(opts, idx, only_T2=False):
    """prostate segmentation with model loaded from checkpoints"""
    T2 = np.expand_dims(np.load('%sT2/sub%03d.npy' % (opts.dir_in, idx)), axis=0)
    if only_T2:
        return np.expand_dims(T2, axis=0)
    else:
        ADC = np.expand_dims(np.load('%sADC/sub%03d.npy' % (opts.dir_in, idx)), axis=0)
        return np.expand_dims(np.concatenate((T2, ADC), axis=0), axis=0)


if __name__ == "__main__":
    args = parse_args()
    args.run_id = args.run_id + args.split
    args.ctx = mx.gpu(args.gpu_id)

    dir_out = args.dir_out + 'run%d/epoch%d' % (args.run_id, args.epoch)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    net = gluon.nn.HybridSequential()
    net.add(load_net(args))
    net.add(Softmax())

    if args.test_model:
        test_model(args, net=net)

    start = time.time()
    # for idx in range(1):
    for idx in range(100):
        net.hybridize()
        im = norm1_v0(load_dat(args, idx + 1, only_T2=args.only_T2), thr=.01)
        out = nd.squeeze(nd.argmax(net(nd.array(im, ctx=args.ctx)), axis=1)).asnumpy()
        out = post_proc(out, area_thr_slice=500)
        print('sub %d, process time: % s' % (idx, time.time() - start))
        if args.plot_pred:
            plot_(out, im[0], invisible=True)
            plt.savefig('%s/sub%03d.png' % (dir_out, idx), bbox_inches='tight', dpi=500)
        if args.save_pred:
            np.save('%s/sub%03d.npy' % (dir_out, idx), out)
        plt.close()
        start = time.time()





