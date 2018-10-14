import numpy as np
from scipy import ndimage
from skimage.util import random_noise
import scipy.ndimage.interpolation as itpl
from skimage.exposure import equalize_adapthist
import scipy.ndimage as ndi
import pylab as plt
from scipy.ndimage.measurements import center_of_mass
from skimage.transform import resize
from utils.utils import *


def center_crop(output_size, img, translation=None):
    """main function"""
    th, tw = output_size
    if translation is None:
        translation_values = [0, 0]
    else:
        translation_values = [np.random.randint(-translation[0], translation[0] + 1),
                              np.random.randint(-translation[1], translation[1] + 1)]
    w, h = img.shape[-2:]
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[...,
           i + translation_values[0]: h + translation_values[0] - i,
           j + translation_values[1]: w + translation_values[1] - j]


def rand_fliplr(obj):
    """Flip left right randomly"""
    if np.random.rand() > .5:
        return np.flip(obj.im, axis=-1)
    else:
        return obj.im


def rand_flipud(obj):
    """Flip up down randomly"""
    if np.random.rand() > .5:
        return np.flip(obj.im, axis=-2)
    else:
        return obj.im


def rand_rot(obj):
    """rotate images with a random angle"""
    if np.random.rand() > .5:
        angle = np.random.rand() * obj.max_angle * np.random.choice([1, -1], 1)[0]
        obj.im = ndimage.rotate(obj.im, angle=angle, axes=[-2, -1], reshape=False)
        if obj.label_in_last_channel:
            obj.im[:, -1] = (obj.im[:, -1] > .5) * 1
    return obj.im


def concat_coor_maps(obj):
    """Concatenate inputs with coordinate maps as additional channels"""
    obj.coor_maps = np.tile(obj.coor_maps[0], (obj.im.shape[0], 1, 1, 1, 1))
    return np.concatenate((obj.coor_maps[:, :, :obj.im.shape[2]], obj.im), axis=1)  # maintain that the last channel is label


def rand_scale(obj):
    """Zoom in and out"""
    org_size = obj.im.shape[-1]
    for i in range(obj.im.shape[0]):
        if np.random.rand() < .5:
            continue
        zoom_ratio = np.round((1 + np.random.choice([-1, 1], 1) * (np.random.rand() * 0.2))[0], 2)
        # zoom_ratio = np.round((1 + (np.random.rand() * 0.7)), 2)
        zoom = [1, zoom_ratio, zoom_ratio]
        for j in range(obj.im.shape[1]):
            # pad_mode = 'constant' if j == (obj.im.shape[1]-1) else 'reflect'
            pad_mode = 'edge'
            order = 0 if j == obj.im.shape[1] else 3
            tmp = ndi.zoom(obj.im[i, j], zoom=zoom, order=order, prefilter=True)
            if zoom_ratio < 1:  # zoom out
                pad_width = (org_size - tmp.shape[-1]) // 2
                obj.im[i, j] = np.pad(tmp, pad_width=((0, 0),
                                                      (pad_width, org_size - pad_width - tmp.shape[-2]),
                                                      (pad_width, org_size - pad_width - tmp.shape[-1])), mode=pad_mode)
                # if tmp.shape[-1] > org_size:  # if the size after padded is diff
                #     obj.im[i, j] = crop_center(tmp, org_size, org_size)
                # else:
                #     obj.im[i, j] = tmp
            else:  # zoom in
                obj.im[i, j] = crop_center(tmp, org_size, org_size)
    return obj.im


def drop_out(obj):
    """Randomly drop voxels"""
    if np.random.rand() > .5:
        obj.im[:, :-1] = random_noise(obj.im[:, :-1], mode='pepper')
    return obj.im


def contrast_norm(obj):
    """Enhance or worsen the constrast"""
    return obj.im


def deform(obj):
    """Deform objects"""
    return obj.im


def modify_values(obj):
    """Add, substract by random values"""
    if np.random.rand() > .5:
        mode = 'speckle'
        obj.im[:, :-1] = random_noise(obj.im[:, :-1], mode=mode, clip=True,
                                        mean=np.random.rand() * .5, var=np.random.rand() * 1e-2)
    return obj.im


def Gauss_blur(obj):
    """Blurring"""
    if np.random.rand() > .5:
        sigma = np.random.rand()
        obj.im[:, :-1] = ndi.filters.gaussian_filter(obj.im[:, :-1], sigma)
    return obj.im


def add_noise(obj):
    """Add Gaussian noise"""
    if np.random.rand() > .5:
        # mode = np.random.choice(['gaussian', 'poisson'], 1)[0]
        mode = 'gaussian'
        if mode is 'gaussian':
            obj.im[:, :-1] = random_noise(obj.im[:, :-1], mode=mode,
                                            mean=0, var=np.random.rand() * 1e-2, clip=True)
    return obj.im


def rand_translate(obj):
    """Random translation in x and y dimension"""
    if np.random.rand() > .7:
        shift = list(np.random.choice(np.arange(int(obj.im.shape[-1] * obj.translate_ratio)), 2))
        mode = np.random.choice(['constant', 'nearest'], 1)[0]
        for i in range(obj.im.shape[0]):
            for c in range(obj.im.shape[1]):
                obj.im[:, c] = itpl.shift(obj.im[i, c], [0, ] + shift, mode=mode, order=0)
    return obj.im


def sample_slices(obj):
    """Sampling slices"""
    nsee = obj.nsee
    if not obj.is_val:
        im = np.zeros((obj.im.shape[0], obj.im.shape[1], obj.zdim) + (obj.im.shape[3:]))
    for i in range(obj.im.shape[0]):
        tmp = remove_empty_slices(obj.im[i])
        idx = find_prostate_slices(tmp[-1])
        # strides = [3, 4, 5, 7, 11, 13, 17, 19]
        strides = [5, 7, 9]
        # strides = [9, ]
        if obj.is_val:
            ims = []
            idx_list = []
            for st in strides:
                idx_ = gen_idx(tmp.shape[1], obj.zdim, st)
                [idx_list.append(idx) for idx in idx_ if idx not in idx_list]
            [ims.append(tmp[np.newaxis, :, idx_[0]: idx_[1]]) for idx_ in idx_list]

            im = np.concatenate((*ims, ), axis=0)
            idx_array = []
            [idx_array.append(np.arange(idx_list_[0], idx_list_[1])[np.newaxis]) for idx_list_ in idx_list]
            idx_array = np.concatenate((*idx_array,), axis=0)

            # TTA - flipping left right
            im_fliplr = np.flip(im, axis=-1)
            idx_array_fliplr = idx_array * - 1

            # TTA - flipping up down
            im_flipud = np.flip(im, axis=-2)
            idx_array_flipud = idx_array * - 1 - 100

            # # TTA - rotation
            # im_rot = im
            # im_rot[:, 0] = ndimage.rotate(im_rot[:, 0], angle=-5, axes=[-2, -1], reshape=False)
            # idx_array_rot = idx_array * -1 - 100

            # Concatenate with original image
            im = np.concatenate((im, im_fliplr, im_flipud), axis=0)
            idx_array = np.concatenate((idx_array, idx_array_fliplr, idx_array_flipud), axis=0)

            idx_array_sample = np.zeros((1, ) + (im.shape[1:])) - 999
            idx_array_sample[0, -1, 0, :idx_array.shape[0], :idx_array.shape[1]] = idx_array
            im = np.concatenate((im, idx_array_sample), axis=0)
        else:
            mid_len = round(np.arange(idx[nsee], idx[-nsee+1]).__len__() * obj.mid_len_ratio)
            mid_idx = np.random.choice(np.arange(idx[nsee], idx[-nsee+1]), min(mid_len, obj.zdim - nsee*2), replace=True)

            out_idx = np.random.choice(np.setdiff1d(np.arange(tmp.shape[1]), idx), obj.zdim - len(mid_idx) - nsee*2, replace=True)
            apex_idx = np.random.choice(idx[:nsee], nsee)
            base_idx = np.random.choice(idx[-nsee:], nsee)
            idx = np.sort(np.concatenate((out_idx, apex_idx, mid_idx, base_idx)))
            im[i] = tmp[:, idx]
    return im


def crop_and_resize(obj):
    """Randomly crop and resize to crop_size"""
    # im = np.zeros((obj.im.shape[0], obj.im.shape[1], 20, 350, 350))
    # im[:, :, 2: 18, 150: 200, 150: 200] = 1
    n_crops = obj.crop_size_list.__len__()
    im = obj.im
    # crop_vols = np.zeros((im.shape[0], im.shape[1] * n_crops, obj.zdim, obj.crop_size, obj.crop_size))
    crop_vols = np.zeros((im.shape[0], (im.shape[1]-1) * n_crops + 1, obj.zdim, obj.crop_size, obj.crop_size))

    for i in range(im.shape[0]):
        # Compute center, height and width
        if obj.is_val:
            if i == im.shape[0]-1:  # last image contains slides idx
                continue
            center = np.array(center_of_mass(im[i, -1])[1:]).astype('int')
            h, w = im.shape[-2], im.shape[-1]
        else:
            center = np.random.multivariate_normal(center_of_mass(im[i, -1])[1:], obj.cov).astype('int')
            scaling_factor = np.random.normal(obj.mean_scaling, .1, 2)
            scaling_factor = [1 if scaling_factor[j] > 1 else scaling_factor[j] for j in range(2)]
            h = int(im.shape[-2] * scaling_factor[0])
            w = int(im.shape[-1] * scaling_factor[1])
        # Random zoom in
        if not ((h == im.shape[-2]) and (w == im.shape[-1])):
            im_cr = crop_center(im[i], h, w)
            for j in range(im.shape[1]):
                order = 0 if j == im.shape[1]-1 else 3
                u = np.transpose(im_cr[j], axes=(1, 2, 0))
                im[i, j] = np.transpose(resize(u, [im.shape[-2], im.shape[-1]], order=order, mode='constant'), axes=(2, 0, 1))
        # Crop and resize multi-scales
        for (crop_size_idx, crop_size_) in enumerate(obj.crop_size_list):
            crop_vol = crop_center(im[i], crop_size_, crop_size_, center)
            for j in range(im.shape[1]-1):
                order = 0 if j == im.shape[1] - 1 else 3
                u = np.transpose(crop_vol[j], axes=(1, 2, 0))
                if obj.to_norm:
                    u = norm_01(u, obj.norm_thr)
                if u.shape[0] == obj.crop_size:
                    crop_vols[i, j * n_crops + crop_size_idx] = np.transpose(u, axes=(2, 0, 1))
                else:
                    crop_vols[i, j * n_crops + crop_size_idx] = \
                        np.transpose(resize(u, [obj.crop_size, obj.crop_size], order=order, mode='constant'), axes=(2, 0, 1))
        crop_vols[i, -1] = crop_center(im[i, -1], obj.crop_size, obj.crop_size, center)
    if obj.is_val:
        crop_vols[-1, -1, 0] = im[-1, -1, 0, :obj.crop_size, :obj.crop_size]
    return crop_vols


def swap_channels(obj):
    """Randomly swap T2 and ADC channels"""
    if obj.im.shape[1] == 3:
        if not obj.is_val and (np.random.rand() > .5):
            obj.im = obj.im[:, [1, 0, 2]]
    return obj.im


def sample_slices_v0(obj):
    """Sampling slices"""
    if not obj.is_val:
        im = np.zeros((obj.im.shape[0], obj.im.shape[1], obj.zdim) + (obj.im.shape[3:]))
    for i in range(obj.im.shape[0]):
        tmp = remove_empty_slices(obj.im[i])
        idx = find_prostate_slices(tmp[-1])
        if obj.is_val:
            ims = []
            idx_list = []
            idx_check = np.arange(tmp.shape[1])
            while len(idx_check) > 0:
                idx = np.sort(np.random.choice(tmp.shape[1], obj.zdim, replace=False))
                idx_check = np.setdiff1d(idx_check, idx)
                ims.append(obj.im[i:i+1, :, idx])
                idx_list.append(idx[np.newaxis])
            im = np.concatenate((*ims, ), axis=0)
            idx_list = np.concatenate((*idx_list,), axis=0)
            idx_list_sample = np.zeros((1, ) + (im.shape[1:])) - 1
            idx_list_sample[0, -1, 0, :idx_list.shape[0], :idx_list.shape[1]] = idx_list
            im = np.concatenate((im, idx_list_sample), axis=0)
        else:
            mid_len = round(np.arange(idx[3], idx[-3]).__len__() * obj.mid_len_ratio)
            mid_idx = np.random.choice(np.arange(idx[3], idx[-3]), min(mid_len, obj.zdim - 6), replace=True)

            out_idx = np.random.choice(np.setdiff1d(np.arange(tmp.shape[1]), idx), obj.zdim - len(mid_idx) - 6, replace=True)
            apex_idx = np.random.choice(idx[:3], 3)
            base_idx = np.random.choice(idx[-3:], 3)
            idx = np.sort(np.concatenate((out_idx, apex_idx, mid_idx, base_idx)))
            im[i] = obj.im[i:i+1, :, idx]
    return im


def remove_empty_slices(im):
    """Remove empty slices, only applied when batch_size is 1"""
    s = np.squeeze(np.sum(im[-2:], axis=(-1, -2, -4)))
    im = im[:, np.argwhere(s > 0)[:, 0]]
    return im


def gen_idx(len_, zdim, stride):
    """Generate index of each sliding"""
    idx_list = []
    start_ = 0
    end_ = 0
    while end_ < len_:
        end_ = start_ + zdim
        if end_ > len_:
            end_ = len_
            start_ = end_ - zdim
        idx_list.append([start_, end_])
        start_ += stride
    return idx_list


def find_prostate_slices(lab):
    """Return indices of slices with prostate label"""
    s = np.squeeze(np.sum(lab, axis=(-1, -2)))
    return np.argwhere(s > 0)[:, 0]


def crop_center(img, size_x, size_y, center=None):
    """2D center crop for ND image"""
    y, x = img.shape[-2:]
    if center is None:
        center = [y//2, x//2]
    start_x = center[1] - size_x//2
    start_y = center[0] - size_y//2
    return img[..., start_y: start_y + size_y, start_x: start_x + size_x]
