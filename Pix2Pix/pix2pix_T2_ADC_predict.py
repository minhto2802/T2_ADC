import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import inspect

from datetime import datetime
import time

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, BatchNorm, LeakyReLU, Flatten, \
    HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
from mxboard import SummaryWriter
import logging

'''SET TRAINING PARAMETERS'''
run_id = 22
fold = 1
gpu_id = 0

no_mask = False
prostate_segmentation = False
no_loss_weights = True
unmasked_T2 = False

resumed_epochs = 380
use_gpu = True
ctx = mx.gpu(gpu_id) if use_gpu else mx.cpu()
norm_data = True

'''DOWNLOAD AND PREPROCESS DATASET'''
dataset = 'AMC_GAN_T2_ADC'

dir_in = 'F:\Minh\mxnet\projects\cancer_segmentation\inputs/'
dir_out = r"F:/BACKUPS/%s/outputs/run%d/" % (dataset, run_id)
dir_out_checkpoints = r"%s/checkpoints/" % dir_out
dir_out_sw = r"%s/logs/" % dir_out

input_file_suffix = ''


def make_dirs(mdir):
    """making directories"""
    if not os.path.exists(mdir):
        os.makedirs(mdir)


def transform_data(x):
    """transform data into MXNet format (NCHW)"""
    if x.ndim == 4:
        return x.transpose((3, 2, 0, 1))
    elif x.ndim == 3:
        return np.expand_dims(x.transpose((2, 0, 1)), axis=1)


def norm1(x):
    """normalize data to 0 mean and 1 std"""
    x_norm = np.zeros(x.shape)
    thr = .1
    for i in range(x.shape[-1]):
        for j in range(x.shape[-2]):
            tmp = np.sort(np.ndarray.flatten(x[:, :, j, i]))
            low_lim = tmp[int(np.around(tmp.shape[0] * thr))]
            high_lim = tmp[int(np.around(tmp.shape[0] * (1 - thr)))]
            tmp = tmp[(tmp >= low_lim) & (tmp <= high_lim)]
            if len(tmp) == 0:
                print(i)
            # tmp1 = x[:, :, j, i]
            # tmp1[tmp1 > high_lim] = high_lim
            # tmp1[tmp1 < low_lim] = low_lim
            # x[:, :, j, i] = tmp1
            x_norm[:, :, j, i] = (x[:, :, j, i] - np.mean(tmp)) / np.std(tmp)
    return x_norm


def norm2(x):
    """normalize data to 0 -> 1 per channel, per slice (for numpy)"""
    x_norm = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_norm[i, j, :, :] = norm_01(x[i, j, :, :])
    return x_norm


def norm3(x):
    """normalize data to 0 -> 1 per channel, per slice (for mxnet ndarray)"""
    x_norm = mx.nd.zeros(x.shape, ctx=x.context)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_norm[i, j, :, :] = norm_01(x[i, j, :, :])
    return x_norm


def norm_01(x):
    """normalize to 0->1"""
    return (x - x.min()) / (x.max() - x.min())


# Define Unet generator skip block
class UnetSkipUnit(HybridBlock):
    """Unet skip unit"""
    def __init__(self, inner_channels, outer_channels, inner_block=None,
                 innermost=False, outermost=False, use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=.2)
            en_norm = BatchNorm(momentum=.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation='tanh')]
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        """forward"""
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)


# Define Unet generator
class UnetGenerator(HybridBlock):
    """generator"""
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        """forward"""
        return self.model(x)


def set_network():
    """set networks_"""
    # Pixel2pixel networks_
    netG = UnetGenerator(in_channels=1, num_downs=5)

    # Initialize parameters
    netG.load_params('%s/netG-%04d' % (dir_out_checkpoints, resumed_epochs), ctx=ctx)

    return netG


########################################################################################################################
# Set loggers
# logger = logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
log_file = 'log.txt'
logging.basicConfig(filename='%s/%s' % (dir_out, log_file),
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
# coloredlogs.install()

# Log all the passed function parameters
frame = inspect.currentframe()
args, _, _, values = inspect.getargvalues(frame)
logger.info('function name "%s"' % inspect.getframeinfo(frame)[2])
for i in args:
    logger.info("    %s = %s" % (i, values[i]))
logger.info('Output directory: %s' % dir_out)

# load data
folds = np.load(r'%sfolds.npy' % dir_in)
caseID = np.load(r'%scaseID.npy' % dir_in)[0]
im = transform_data(np.load('%sim%s.npy' % (dir_in, input_file_suffix)))
mask = transform_data(np.load('%smask%s.npy' % (dir_in, input_file_suffix)))

# remove cancer ROIs (unnecessary for this task)
mask[mask == 2] = 1
if prostate_segmentation:
    im[:, 1] = mask[:, 0]
elif unmasked_T2:
    im[:, 1] = im[:, 1] * mask[:, 0]
else:
    im = im * mask

# loss_weights = mask
# loss_weights[mask == 0] = .1  # small penalty for the loss in background

if no_mask:
    mask[mask == 0] = 1

# normalize data
if norm_data:
    logger.info('Normalizing data across the whole dataset... ')  # per channel
    im = norm2(im)

# get network
netG = set_network()

fake_out = netG(mx.nd.array(im[:, 0, None], ctx=ctx)) * mx.nd.array(mask, ctx=ctx)

im[:, 1, None] = fake_out.asnumpy()
np.save('%s/fake_im_fold%d.npy' % (dir_in, fold), im)

