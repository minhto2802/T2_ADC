""""Function definitions"""
import numpy as np
from mxnet import nd
from mxnet.gluon import loss
from mxnet.gluon.loss import _reshape_like, _apply_weighting
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, generate_binary_structure, binary_dilation


def dice_wp(preds, labels):
    """Compute Dice per case between prediction and label"""
    smooth = 1.
    if preds.shape != labels.shape:
        preds = preds.argmax(axis=1)
    dice = nd.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        dice[i] = ((2 * (labels[i] * preds[i]).sum()) + smooth) / (labels[i].sum() + preds[i].sum() + smooth)
    return dice


class DiceLoss(loss.Loss):
    """correlation loss"""
    def __init__(self, axis=[0, 1, 2], weight=1., batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._batch_axis = batch_axis
        self.smooth = 1.

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Forward"""
        label = nd.one_hot(label, depth=2).transpose((0, 4, 1, 2, 3))
        intersection = F.sum(label * pred, axis=self._axis, exclude=True)
        union = F.sum(label + pred, axis=self._axis, exclude=True)
        dice = (2.0 * F.sum(intersection, axis=1) + self.smooth) / (F.sum(union, axis=1) + self.smooth)
        # return F.log(1 - dice)
        # return 1-dice
        # return F.exp(-dice)
        return 1 - dice.sum() / np.prod(dice.shape)


class CELoss(loss.Loss):
    """Cross Entropy Loss"""
    def __init__(self, axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super(CELoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Forward"""
        pred = F.log(pred)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


def norm_0mean(x):
    x_norm = x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                mu = x[i, j, k].mean()
                std = x[i, j, k].std()
                x_norm[i, j, k] = (x[i, j, k] - mu)/std
    return x_norm


def thr_by_size(x, thr=None):
    if thr is None:
        thr = np.prod(x.shape) * 1e-2
    blobs_labels, num_labels = measure.label(x, background=0, return_num=True)
    label_vols = np.zeros((num_labels + 1, ))
    for j in range(1, num_labels + 1):
        label_vols[j] = blobs_labels[blobs_labels == j].sum()
        x[blobs_labels == j] = 0 if label_vols[j] < thr else 1
        # blobs_labels[blobs_labels == j] = 0 if label_vol < thr else j
    # x[blobs_labels != label_vols.argmax()] = 0
    return x


def get_blobs_max_length(x):
    blobs_labels, num_labels = measure.label(x, background=0, return_num=True)
    label_length = np.zeros((num_labels + 1, ))
    for j in range(1, num_labels + 1):
        tmp = np.zeros(blobs_labels.shape)
        tmp[blobs_labels == j] = 1
        label_length[j] = (tmp.sum(axis=(-2, -1)) > 0).sum()
    x[blobs_labels != label_length.argmax()] = 0
    return x


def fill_holes(x):
    # struct = generate_binary_structure(3, 2)
    # return binary_closing(x, structure=struct)
    for i in range(x.shape[0]):
        x[i] = binary_closing(x[i], structure=np.ones((5, 5)))
    return x


def post_proc(x):
    # return fill_holes(thr_by_size(x))
    return fill_holes(thr_by_size(get_blobs_max_length(x)))
    # return fill_holes(x)
    # return thr_by_size(x)


def norm_01(im, thr=.98):
    """Normalize image to 0-1 while thresholding the intensity values to within range [1-thr, thr]"""
    s1d = im.flatten()
    s1d_sort = np.sort(s1d)
    uthr = s1d_sort[int(np.floor(s1d_sort.shape[0] * thr))]
    lthr = s1d_sort[int(np.floor(s1d_sort.shape[0] * (1 - thr)))]

    s1d[s1d >= uthr] = np.median(s1d[(lthr < s1d) & (s1d < uthr)])
    s1d[s1d <= lthr] = np.median(s1d[(lthr < s1d) & (s1d < uthr)])
    s1d = (s1d - np.min(s1d)) / (np.max(s1d) - np.min(s1d))

    im_norm = np.reshape(s1d, im.shape)
    return im_norm
