# coding=utf-8
"""Functions collection"""
from T2_ADC.utils.augmentations import *
import numpy as np
import inspect
from mxnet import nd, gluon, initializer
from outliers import smirnov_grubbs as grubbs
from T2_ADC.networks.pix2pix_nets import UnetGenerator, Discriminator
from T2_ADC.networks.dmnet_gluon_init_2d import Init, DenseMultipathNet
import pylab as plt
import os
from scipy import ndimage


def augmenter(data, patch_size, offset, aug_type=1, aug_methods='d', r_int_lim=30, random_crop=True):
    """implement augmentation"""
    if aug_methods == 'd':  # default
        aug_methods = {'random_rota90': random_rot90,
                       'flip': random_flip}

    r_int = np.random.randint(-r_int_lim, r_int_lim) * np.random.rand()
    if aug_type == 0:  # augmentation per sample
        for ii in range(data.shape[0]):
            aug_names = list(aug_methods.keys())
            for aug_time in np.arange(len(aug_names)):
                aug = aug_names[np.random.randint(len(aug_names))]
                if np.random.rand() > .5:
                    data[ii] = aug_methods[aug](data[ii], r_int)
                aug_names.remove(aug)
    elif aug_type == 1:  # augmentation per batch
        aug_names = list(aug_methods.keys())
        for aug_time in np.arange(len(aug_names)):
            aug = aug_names[np.random.randint(len(aug_names))]
            if np.random.rand() > .5:
                data = aug_methods[aug](data, r_int)
            aug_names.remove(aug)

    if random_crop:
        return my_crop(data, patch_size, offset, True)
    else:
        return data


def my_crop(im, patch_size, offset, random_crop=False):
    """Crop images"""
    centroid = np.array([im.shape[-1]/2, im.shape[-2]/2], 'uint8')

    if random_crop:
        if inspect.isclass(im.dtype):
            im_new = nd.zeros((im.shape[0], im.shape[1], patch_size, patch_size))
        else:
            im_new = np.zeros((im.shape[0], im.shape[1], patch_size, patch_size))
        for i in range(im.shape[0]):
            translation_vals = np.array([np.random.randint(offset[0], offset[1]),
                                         np.random.randint(offset[2], offset[3])])
            centroid_new = centroid + translation_vals
            r = int(patch_size/2)
            im_new[i] = im[i, :, centroid_new[0] - r: centroid_new[0] + r, centroid_new[1] - r: centroid_new[1] + r]
    else:
        centroid_new = centroid
        r = int(patch_size / 2)
        im_new = im[:, :, centroid_new[0] - r: centroid_new[0] + r, centroid_new[1] - r: centroid_new[1] + r]
    return im_new


def norm2(x, return_stats=False, mask=None):
    """normalize data to 0 -> 1 per channel, per slice (for numpy)"""
    x_norm = np.zeros(x.shape)
    if return_stats:
        stats = np.zeros((x.shape[0], x.shape[1], 2, 2)) # 3rd dimension: pre_norm, post_norm
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if mask is None:
                x_norm[i, j] = norm_01(x[i, j])
            else:
                x_norm[i, j] = norm_01(x[i, j], mask[i, 0])
            if return_stats:
                stats[i, j, 0, 0] = x[i, j].flatten().mean()
                stats[i, j, 0, 1] = x[i, j].flatten().std()
                stats[i, j, 1, 0] = x_norm[i, j].flatten().mean()
                stats[i, j, 1, 1] = x_norm[i, j].flatten().std()
    if return_stats:
        return x_norm, stats
    else:
        return x_norm


def norm_01(x, mask=None):
    """normalize to 0->1"""
    if mask is None:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - x[mask == 1].min()) / (x[mask == 1].max() - x[mask == 1].min())


def norm1_v0(x, thr=.05):
    """Adapted from prostate_segmenetation/my_func.py, changes were made in dimension slicing"""
    x_norm = np.zeros(x.shape)
    # thr = .005
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                tmp = np.sort(np.ndarray.flatten(x[i, j, k]))

                low_lim = tmp[int(np.around(tmp.shape[0] * thr))]
                high_lim = tmp[int(np.around(tmp.shape[0] * (1 - thr)))]
                tmp = tmp[(tmp >= low_lim) & (tmp <= high_lim)]

                if len(tmp) == 0:
                    print(i)
                # tmp1 = np.zeros(x[:, :, k, j, i].shape) + x[:, :, k, j, i]
                # tmp1[tmp1 > high_lim] = high_lim
                # tmp1[tmp1 < low_lim] = low_lim
                # x[:, :, k, j, i] = tmp1
                x_norm[i, j, k] = (x[i, j, k] - np.mean(tmp)) / (np.std(tmp) + 1e-8)
                # print(np.sum(x_norm))
                # print
    return x_norm


def norm1(x, mask=None, return_stats=False, othr=.05):
    """normalize data to 0 mean, 1 std, with outliers exclusion, per slice"""
    x_norm = np.zeros(x.shape)
    if return_stats:
        stats = np.zeros((x.shape[0], x.shape[1], 3, 2))  # 3rd dimension: pre_grubbs, post_grubs, post_norm
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if mask is None:
                tmp = grubbs.test(x[i, j].flatten(), alpha=othr)
            elif mask[i, 0].sum() > 0:
                tmp = grubbs.test(x[i, j][mask[i, 0] == 1], alpha=othr)
            else:
                tmp = grubbs.test(x[i, j].flatten(), alpha=othr)
            x_norm[i, j] = (x[i, j] - tmp.mean()) / (tmp.std())
            if return_stats:
                stats[i, j, 0, 0] = x[i, j].flatten().mean()
                stats[i, j, 0, 1] = x[i, j].flatten().std()
                stats[i, j, 1, 0] = tmp.mean()
                stats[i, j, 1, 1] = tmp.std()
                stats[i, j, 2, 0] = x_norm[i, j].flatten().mean()
                stats[i, j, 2, 1] = x_norm[i, j].flatten().std()
    if return_stats:
        return x_norm, stats
    else:
        return x_norm


def remove_outliers(x, std_degree=3, return_outliers=False):
    """find outliers in 2D array"""
    sigma = np.std(x)
    mu = np.mean(x)
    if not return_outliers:
        return x[(x > (mu - std_degree * sigma)) & (x < (x + std_degree * sigma))]
    else:
        outliers = x[(x <= (mu - std_degree * sigma)) | (x >= (x + std_degree * sigma))]
        x = x[(x > (mu - std_degree * sigma)) & (x < (x + std_degree * sigma))]
        return x, outliers


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


def norm3(x):
    """normalize data to 0 -> 1 per channel, per slice (for mxnet ndarray)"""
    x_norm = mx.nd.zeros(x.shape, ctx=x.context)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_norm[i, j, :, :] = norm_01(x[i, j, :, :])
    return x_norm


def create_iterators(im_arr, batch_size):
    """put data into iterator"""
    return mx.io.NDArrayIter(data=[im_arr[:, 0, None], im_arr[:, 1, None], im_arr[:, 2, None]],
                             batch_size=batch_size)  # starred expression to unpack a list


def visualize(img_arr, to_plot=True):
    """viz"""
    if to_plot:
        plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.axis('off')
    else:
        return nd.array(((img_arr.asnumpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).astype(np.uint8))


def preview_train_data():
    """preview data"""
    img_in_list, img_out_list = train_data.next().data
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        visualize(img_in_list[i])
        plt.subplot(2, 4, i + 5)
        visualize(img_out_list[i])
    plt.show()


def param_init(param, ctx):
    """init param"""
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        # Initialize gamma from normal distribution with mean 1 and std 0.02
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))


def network_init(net, ctx):
    """init all network params"""
    for param in net.collect_params().values():
        param_init(param, ctx)


def set_network(ctx, lr, beta1, optimizer):
    """set networks_"""
    # Pixel2pixel networks_
    netG = DenseMultipathNet(Init(num_fpg=-1))
    # netG = UnetGenerator(in_channels=1, num_downs=5)
    netD = Discriminator(in_channels=2)

    # Initialize parameters
    # network_init(netG, ctx)
    netG.initialize(init=initializer.Xavier(magnitude=2), ctx=ctx)
    network_init(netD, ctx)

    # trainer for the generator and the discriminator
    if optimizer == 'rmsprop':
        trainerG = gluon.Trainer(netG.collect_params(), optimizer, {'learning_rate': lr})
        trainerD = gluon.Trainer(netD.collect_params(), optimizer, {'learning_rate': lr})
    else:
        trainerG = gluon.Trainer(netG.collect_params(), optimizer, {'learning_rate': lr, 'beta1': beta1})
        trainerD = gluon.Trainer(netD.collect_params(), optimizer, {'learning_rate': lr, 'beta1': beta1})

    return netG, netD, trainerG, trainerD


def set_network_CycleGAN(ctx, lr, beta1, optimizer):
    """set CycleGAN networks"""
    # Pixel2pixel networks_
    # netG = UnetGenerator(in_channels=1, num_downs=5)
    netGY = DenseMultipathNet(Init(num_fpg=-1))
    netGX = DenseMultipathNet(Init(num_fpg=-1))
    netDY = Discriminator(in_channels=2)
    netDX = Discriminator(in_channels=2)

    # Initialize parameters
    # network_init(netG, ctx)
    netGY.initialize(init=initializer.Xavier(magnitude=2), ctx=ctx)
    netGX.initialize(init=initializer.Xavier(magnitude=2), ctx=ctx)
    network_init(netDY, ctx)
    network_init(netDX, ctx)

    # trainer for the generator and the discriminator
    if optimizer == 'rmsprop':
        trainerGY = gluon.Trainer(netGY.collect_params(), optimizer, {'learning_rate': lr})
        trainerGX = gluon.Trainer(netGX.collect_params(), optimizer, {'learning_rate': lr})
        trainerDY = gluon.Trainer(netDY.collect_params(), optimizer, {'learning_rate': lr})
        trainerDX = gluon.Trainer(netDX.collect_params(), optimizer, {'learning_rate': lr})
    else:
        trainerGY = gluon.Trainer(netGY.collect_params(), optimizer, {'learning_rate': lr, 'beta1': beta1})
        trainerGX = gluon.Trainer(netDX.collect_params(), optimizer, {'learning_rate': lr, 'beta1': beta1})
        trainerDY = gluon.Trainer(netGY.collect_params(), optimizer, {'learning_rate': lr, 'beta1': beta1})
        trainerDX = gluon.Trainer(netDX.collect_params(), optimizer, {'learning_rate': lr, 'beta1': beta1})

    return netGY, netGX, netDY, netDX, trainerGY, trainerGX, trainerDY, trainerDX


class ImagePool:
    """IMAGE POOL FOR DISCRIMINATOR"""
    # We use history image pool to help discriminator memorize history errors
    # instead of just comparing current real input and fake output.
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        ret_imgs = []
        for i in range(images.shape[0]):
            image = nd.expand_dims(images[i], axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                ret_imgs.append(image)
            else:
                p = nd.random_uniform(0, 1, shape=(1, )).asscalar()
                if p > .5:
                    random_id = nd.random_uniform(0, self.pool_size -1, shape=(1, )).astype(np.uint8).asscalar()
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    ret_imgs.append(tmp)
                else:
                    ret_imgs.append(image)
        ret_imgs = nd.concat(*ret_imgs, dim=0)
        return ret_imgs


def facc(label, pred):
    """compute pixel accuracy"""
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


class SmoothL1Loss(gluon.loss.Loss):
    """Smooth L1 loss"""
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        """Forward"""
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)


class Softmax(gluon.HybridBlock):
    """"Softmax"""
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        """Forward"""
        return F.softmax(x, axis=1)


def post_proc(pred, area_thr_slice=200):
    def get_area(label, nb_label):
        return np.sum(label == nb_label)

    label_im, nb_labels = ndimage.label(pred)

    # remove small blobs in each slice
    for j in np.arange(label_im.shape[0]):
        label_slice, nb_labels_slice = ndimage.label(label_im[j])
        for i in np.arange(1, nb_labels_slice + 1):
            if get_area(label_slice, i) < area_thr_slice:
                label_slice[label_slice == i] = 0
        label_im[j] = label_slice

    # estimate centroids
    for j in np.arange(label_im.shape[0]):
        label_slice, nb_labels_slice = ndimage.label(label_im[j])
        centroids = np.zeros((nb_labels_slice, 2))
        for i in np.arange(1, nb_labels_slice + 1):
            centroids[i - 1, :] = ndimage.measurements.center_of_mass(label_slice == i)

        expected_centroid = [pred.shape[i] / 2 for i in np.arange(-2, 0)]

        distance = np.square(np.sum((centroids - expected_centroid) ** 2, axis=1))
        if distance.size > 0:
            label_im[j] = label_slice == (np.argmin(distance) + 1)

    # fill holes
    pred = ndimage.morphology.binary_fill_holes(label_im)

    return pred