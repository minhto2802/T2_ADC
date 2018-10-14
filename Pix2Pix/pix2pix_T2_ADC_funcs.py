"""A collection of function definitions"""
import pylab as plt
# import mxnet as mx
import collections
import scipy.ndimage as ndimage
import numpy as np
import os
from itertools import permutations
from skimage import morphology


def is_np_array(I):
    """check if I is mx-ndarray of np-ndarray"""
    return isinstance(I, collections.abc.Iterable)


def to_np_array(I):
    """convert mx-ndarray to np-ndarray"""
    if not is_np_array(I):  # check if input is numpy array
        return I.asnumpy()
    else:
        return I


def visualize(im, mask, only_ROIs=True):
    """show im and cancer contour"""
    im = to_np_array(im)
    mask = to_np_array(mask)

    # only show cancer region
    if only_ROIs:
        mask[mask < mask.max()] = 0

    plt.imshow(im, cmap='gray'), plt.hold
    plt.contour(mask)


def crop_cancer_ROIs(im, mask, label, max_translation_range=.3, patch_size=64, is_training=True):
    """get a patch containing cancer ROIs with translation (translating the bounding box)"""
    im = to_np_array(im)
    mask = to_np_array(mask)

    if label == 1:  # cancer case
        mask[mask < mask.max()] = 0  # only keep the cancer ROIs
        pad_width = int(patch_size / 2)  # pad_width == 1/2 of patch_size
        if not is_training:
            max_translation_range = 0
    else:
        max_translation_range = .5
        pad_width = int(patch_size)  # pad_width == 1/2 of patch_size

    mask = np.pad(mask, ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=0)
    im = np.pad(im, ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=0)

    # choose one among all tumors
    lbl = ndimage.label(mask)[0]
    random_label_idx = np.random.randint(lbl.max())
    lbl[lbl != (random_label_idx + 1)] = 0

    # get bounding box from label
    bb = np.array(get_bounding_box(lbl))  # y, x order
    bb_dims = np.array([bb[2] - bb[0], bb[3] - bb[1]])

    # find the centroid
    c = np.array([int(np.array([bb[0], bb[2]]).mean()), int(np.array([bb[1], bb[3]]).mean())])  # centroid
    bb_patch = c - int(patch_size/2)
    # c = ndimage.center_of_mass(mask, lbl, list(np.arange(lbl.max()) + 1))[0]  # z, y, x order

    # get translated centroid
    max_translation_values = np.ceil(bb_dims * max_translation_range)
    translation_vector = [np.random.randint(-max_translation_values[0], max_translation_values[0]),
                          np.random.randint(-max_translation_values[1], max_translation_values[1])]
    bb_new = bb_patch + np.array(translation_vector)

    # crop im and mask
    im_cr = np.zeros((im.shape[0], patch_size, patch_size)) + im[:, bb_new[0]: bb_new[0] + patch_size, bb_new[1]: bb_new[1] + patch_size]
    mask_cr = np.zeros((mask.shape[0], patch_size, patch_size)) + mask[:, bb_new[0]: bb_new[0] + patch_size, bb_new[1]: bb_new[1] + patch_size]

    return im_cr, mask_cr


def get_bounding_box(mask):
    """find a single bounding box in an binary image"""
    mask = to_np_array(mask)
    obj = ndimage.find_objects(mask)[-1][-2:]
    return [obj[0].start, obj[1].start, obj[0].stop, obj[1].stop]


def make_dirs(mdir):
    """making directories"""
    if not os.path.exists(mdir):
        os.makedirs(mdir)


class Augmenter:
    """Augmentations"""
    def __init__(self, aug_list, aug_chance=.5, max_dropout_percentage=.5,
                 max_rot=359, exclude=True):
        # super(Augmenter, self).__init__()
        self.aug_list = ['rot90', 'rot', 'add_noise', 'deform', 'hist_norm', 'scale', 'fliplr', 'flipud', 'dropout',
                         'gaussian', ]
        if not exclude:
            self.aug_list = aug_list
        else:
            self.aug_list = [aug for aug in self.aug_list if aug not in aug_list]
        self.aug_dict ={'rot90': self.rot90,
                        'rot': self.rot,
                        'fliplr': self.fliplr,
                        'flipud': self.flipud,
                        'dropout': self.dropout,
                        'add_noise': self.add_noise,
                        'deform': self.deform,
                        'scale': self.scale,
                        'hist_norm': self.hist_norm,
                        'gaussian': self.gaussian_filt}
        self.aug_list_permutations = list(permutations(self.aug_list, self.aug_list.__len__()))
        self.aug_chance = aug_chance
        self.max_rot = max_rot
        self.max_dropout_percentage = max_dropout_percentage

    def fliplr(self):
        """flip left right"""
        self.im = np.flip(self.im, -1)

    def flipud(self):
        """flip up down"""
        self.im = np.flip(self.im, -2)

    def rot(self):
        """rotation at angle angle"""
        r_int = np.random.rand() * self.max_rot
        self.im = ndimage.interpolation.rotate(self.im, r_int, axes=(-2, -1), reshape=False, mode='reflect')

    def rot90(self):
        """randomly rotate images at an angle among [90, 180, 270]"""
        self.im = np.rot90(self.im, k=np.random.randint(1, 4), axes=(-2, -1))

    def add_noise(self):
        """adding Gaussian noise"""
        noise = np.random.random(self.im.shape) * self.im.std() * 3
        self.im = self.im + noise

    def dropout(self):
        """deformation"""
        for ch in range(self.im.shape[0]):
            dropout_percentage = self.max_dropout_percentage * np.random.rand()
            n_dropout = np.floor(np.prod(self.im[ch].shape) * dropout_percentage).astype('int')

            b_loc = np.argwhere(self.im[ch] != 0)
            dropout_loc = np.random.permutation(b_loc.shape[0])[:n_dropout]

            self.im[ch, b_loc[dropout_loc][:, 0], b_loc[dropout_loc][:, 1]] = 0

            self.im[ch] = self.im[ch] * morphology.binary_dilation(self.im[ch].astype('uint8'), morphology.diamond(1.5))

    def gaussian_filt(self):
        """gaussian filter with random sigma"""
        ndimage.gaussian_filter(self.im, sigma=np.random.rand(), output=self.im)

    def deform(self):
        """deformation"""

    def hist_norm(self):
        """random histogram equalization"""

    def scale(self):
        """random histogram equalization"""
        return None

    def get_permuted_aug_list(self):
        """get the list of augmentations with permuted order"""
        return self.aug_list_permutations[np.random.randint(self.aug_list_permutations.__len__())]

    def forward(self, im):
        """perform augmentations"""
        aug_list = self.get_permuted_aug_list()
        self.im = im
        for aug_name in aug_list:
            if np.random.rand() > self.aug_chance:
                self.aug_dict[aug_name]()
        return self.im

