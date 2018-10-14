import numpy as np
from os import getcwd
from os.path import dirname


def load_npy(crop_size=240, test_phase=False, filename=None):
    """Load .npy preprocessed in Matlab"""
    # dir_A = r'%s\ProstateX\data' % dirname(dirname(getcwd()))
    # A = np.expand_dims(np.load('%s/T2_combined_%d_preproc.npy' % (dir_A, crop_size)), axis=1)
    dir_A = 'F:\Minh\projects\PROMISE2012\T2_ADC\im_matched/training/npy/'
    if filename is None:
        # filename = '%s/T2_combined_%d_preproc_histmatch.npy' % (dir_A, crop_size)
        filename = '%s/IMs2d_%d_full.npy' % (dir_A, crop_size)
    A = np.load(filename)
    if np.ndim(A) < 4:
        A = np.expand_dims(np.load(filename), axis=1)
    # A = np.expand_dims(np.load('%s/T2_%d_normed_01_PROMISE.npy' % (dir_A, crop_size)), axis=1)
    if not test_phase:
        # dir_B = dir_A
        # B = np.expand_dims(np.load('%s/ADC_%d_preproc.npy' % (dir_B, crop_size)), axis=1)
        dir_B = r'%s\ProstateX\data' % dirname(dirname(getcwd()))
        B = np.expand_dims(np.load('%s/ADC_%d_full.npy' % (dir_B, crop_size)), axis=1)
        A1 = np.expand_dims(np.load('%s/T2_%d_full.npy' % (dir_B, crop_size)), axis=1)
        A = np.concatenate((A, A1), axis=0)
        return A, B
    else:
        return A


if __name__ == '__main__':
    A, B = load_npy(crop_size=240)
    print(A.shape)
    print(B.shape)
