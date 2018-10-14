# coding=utf-8
"""Generate simulated ADC for PROMISE12 dataset"""
import numpy as np
from mxnet import gluon
import mxnet as mx
from T2_ADC.networks.pix2pix_nets import UnetGenerator
from T2_ADC.networks.dmnet_gluon_init_2d import DenseMultipathNet, Init
from T2_ADC.utils.my_func import norm2, norm1
import os
import pylab as plt

show_im = True

dir_in = r'F:\Workspace\PROMISE12\npy'
im_val = np.load('%s/im_val.npy' % dir_in)
lab_val = np.load('%s/lab_val.npy' % dir_in)

dataset = 'AMC_GAN_T2_ADC'
run_id = 50
epoch = 460
dir_model = r'F:\BACKUPS\AMC_GAN_T2_ADC\outputs\run%d\checkpoints' % run_id
dir_out = '%s/pm12_prediction/' % dir_model

print(dir_out)

to_plot = True
to_predict = True
to_print_stats = False

othr = 1e-5

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

ctx = mx.gpu()

# netG = UnetGenerator(in_channels=1, num_downs=5)
netG = DenseMultipathNet(Init(num_fpg=-1))
netG.load_params('%s/netG-%04d' % (dir_model, epoch), ctx=ctx)
netG.hybridize()
# Set the params
sub_idx = 0
# for sub_idx in range(im_val.shape[0]):
for sub_idx in range(0, 1):
    data = norm1(im_val[sub_idx: sub_idx + 1, 0], mask=lab_val[sub_idx: sub_idx + 1, :], othr=othr)
    data = mx.nd.transpose(mx.nd.array(data, ctx=ctx), (1, 0, 2, 3))
    mask = np.transpose(lab_val[sub_idx: sub_idx + 1, :], (1, 0, 2, 3))

    if to_predict:
        out = netG(data).asnumpy()

        if to_plot:
            for j in range(out.shape[0]):
                plt.subplot(4, int(np.ceil(out.shape[0]/4)), j+1)
                plt.imshow(data[j, 0].asnumpy(), cmap='gray')
                plt.hold('on')
                plt.contour(mask[j, 0])
            plt.figure(2)
            for j in range(out.shape[0]):
                plt.subplot(4, int(np.ceil(out.shape[0]/4)), j+1)
                plt.imshow(out[j, 0], cmap='gray')
                plt.hold('on')
                plt.contour(mask[j, 0])
            if show_im:
                plt.show()
                print()
            # plt.savefig('%ssub_%03d.png' % (dir_out, sub_idx), out)
            # plt.close()

        np.save('%ssub_%03d' % (dir_out, sub_idx), out)


    for j in range(data.shape[0]):
        print(data[j, 0].asnumpy().mean(), data[j, 0].asnumpy().std(), data[j, 0].asnumpy().min(), data[j, 0].asnumpy().max())
