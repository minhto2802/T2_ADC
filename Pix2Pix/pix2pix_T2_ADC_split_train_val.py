# coding=utf-8
# load data
import numpy as np


def norm_01(x, mask=None):
    """normalize to 0->1"""
    if mask is None:
        mask = np.ones(x.shape)
    xmin = x[mask == 1].min()
    xmax = x[mask == 1].max()
    return (x - xmin) / (xmax - xmin)


def norm1(x, mask=None):
    """normalize data to 0 mean and 1 std"""
    if mask is None:
        mask = np.ones(x.shape)
    x_norm = np.zeros(x.shape)
    thr = .1
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = np.sort(x[i, j][mask[i, 0] == 1])
            low_lim = tmp[int(np.around(tmp.shape[0] * thr))]
            high_lim = tmp[int(np.around(tmp.shape[0] * (1 - thr)))]
            tmp = tmp[(tmp >= low_lim) & (tmp <= high_lim)]
            if len(tmp) == 0:
                print(i)
            x_norm[i, j] = (x[i, j] - np.mean(tmp)) / np.std(tmp)
    return x_norm


def norm2(x, mask=None):
    """normalize data to 0 -> 1 per channel, per slice (for numpy)"""
    if mask is None:
        mask = np.ones(x.shape)
    x_norm = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_norm[i, j] = norm_01(x[i, j], mask[i, 0])
    return x_norm


def transform_data(x):
    """transform data into MXNet format (NCHW)"""
    if x.ndim == 4:
        return x.transpose((3, 2, 0, 1))
    elif x.ndim == 3:
        return np.expand_dims(x.transpose((2, 0, 1)), axis=1)


def split_train_val(fold=1, use_fake=False, norm_data=True):
    """main"""
    input_file_suffix = ''
    dir_in = r'F:\Minh\mxnet\projects\cancer_segmentation\inputs/'
    folds = np.load(r'%sfolds.npy' % dir_in)
    caseID = np.load(r'%scaseID.npy' % dir_in)[0]
    c_labels = np.load(r'%sc_labels.npy' % dir_in)[0]

    if not use_fake:
        im = transform_data(np.load('%sim%s.npy' % (dir_in, input_file_suffix)))
    else:
        im = np.load('%sfake_im%s_fold%d.npy' % (dir_in, input_file_suffix, fold))
    mask = transform_data(np.load('%smask%s.npy' % (dir_in, input_file_suffix)))

    # remove cancer ROIs (unnecessary for this task)
    mask_ = np.zeros(mask.shape) + mask
    mask_[mask == 2] = 1
    im = im * mask_  # keeps voxels inside prostate

    # normalize data
    if norm_data:
        print('Normalizing data across the whole dataset... ')  # per channel
        im = norm1(im, mask_) * mask_  # 0 mean, 1 std

    # split dataset
    all_idx = np.arange(caseID.__len__())  # update all_idx
    if fold is not None:
        print('Split dataset into training and validation (fold %d)' % fold)
        # train_amount = int(round(train_percent * im.shape[0]))
        training_set = {}
        training_set['fold%d' % fold] = np.argwhere(folds[0] != fold).transpose().tolist()
        s = set(training_set['fold%d' % fold][0])
        train_idx = [i for i in all_idx if caseID[i] in s]
        val_idx = [i for i in all_idx if caseID[i] not in s]
        im_train = im[train_idx]
        im_val = im[val_idx]
        c_labels_train = c_labels[train_idx]
        c_labels_val = c_labels[val_idx]

    print()
    print('Training set')
    print(im_train.shape)
    print(c_labels_train.shape)
    print(c_labels_train.sum())  # number of cancer cases (1: cancer, 0: benign)

    print()
    print('Validation set')
    print(im_val.shape)
    print(c_labels_val.shape)
    print(c_labels_val.sum())  # number of cancer cases

    return im_train.astype('float32'), im_val.astype('float32'), \
           c_labels_train.astype('uint8'), c_labels_val.astype('uint8'), \
           mask[train_idx].astype('uint8'), mask[val_idx].astype('uint8')


