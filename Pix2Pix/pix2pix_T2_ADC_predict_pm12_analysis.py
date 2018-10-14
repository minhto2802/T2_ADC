# coding=utf-8
"""Inspect the generated ADC of PROMISE12 dataset"""
import numpy as np
import pylab as plt

epoch = 880
run_id = 48
dir_in = r'F:\BACKUPS\AMC_GAN_T2_ADC\outputs\run%d\checkpoints\pm12_prediction' % run_id

for sub_idx in range(1):
    out = np.load('%s/sub_%03d.npy' % (dir_in, sub_idx))
    for j in range(out.shape[0]):
        plt.subplot(4, int(np.ceil(out.shape[0]/4)), j+1)
        plt.imshow(out[j, 0], cmap='gray')
plt.show()
