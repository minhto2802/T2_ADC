from mxnet.gluon.data import dataset, DataLoader
import numpy as np
import mxnet.ndarray as ndarray
from utils.augmentations import *


class MyTransform:
    def __init__(self, im, opts):
        if opts.is_val:
            transforms_list_idx = [0, 14, 1]
        else:
            transforms_list_idx = [0, ] + [opts.transforms_list_idx[j]
                                           for j in np.random.choice(opts.transforms_list_idx.__len__(),
                                                                     opts.transforms_list_idx.__len__(), replace=False)] \
                                  + [14, 1, ]
        self.to_norm = not opts.already_normed
        self.norm_thr = opts.norm_thr
        self.nsee = opts.num_slices_each_end
        self.label_in_last_channel = opts.label_in_last_channel
        self.max_angle = opts.max_angle
        self.coor_maps = opts.coor_maps
        self.is_val = opts.is_val
        self.zdim = opts.zdim
        self.mid_len_ratio = opts.mid_len_ratio
        self.translate_ratio = opts.translate_ratio
        self.mean_scaling = opts.mean_scaling
        self.crop_size_list = opts.crop_size_list
        self.crop_size = opts.crop_size
        self.use_ADC = opts.use_ADC
        self.cov = opts.center_translation_cov
        self.transforms_dict = {
            'sample_slices': sample_slices,  # 0
            'concat_coor_maps': concat_coor_maps,  # concat_coor_maps index must be 0  # 1
            'rand_scale': rand_scale,  # 2
            'drop_out': drop_out,  # 3
            'contrast_norm': contrast_norm,  # 4
            'deform': deform,  # 5
            'modify_values': modify_values,  # 6
            'Gauss_blur': Gauss_blur,  # 7
            'add_Gauss_noise': add_noise,  # 8
            'rand_translate': rand_translate,  # 9
            'rand_rot': rand_rot,  # 10
            'rand_fliplr': rand_fliplr,  # 11
            'center_crop_and_scale': None,  # 12
            'swap_channels': swap_channels,  # 13
            'crop_and_resize': crop_and_resize,  # 14
            'rand_flipud': rand_flipud, # 15
        }
        self.transforms_list = list(self.transforms_dict.keys())
        self.transforms_list = [self.transforms_list[i] for i in transforms_list_idx]
        self.im = im

    def random_transform(self):
        for fn in self.transforms_list:
            self.im = self.transforms_dict[fn](self)
        return self.im


class MyDataset(dataset.Dataset):
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc.

    The i-th sample is defined as `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    *args : one or more dataset-like objects
        The data arrays.
    """
    def __init__(self, opts, *args):
        assert len(args) > 0, "Needs at least 1 arrays"
        self._length = len(args[0])
        self._data = []
        self.opts = opts
        for i, data in enumerate(args):
            assert len(data) == self._length, \
                "All arrays must have the same length; array[0] has length %d " \
                "while array[%d] has %d." % (self._length, i+1, len(data))
            if isinstance(data, ndarray.NDArray) and len(data.shape) == 1:
                data = data.asnumpy()
            self._data.append(data)

    def __getitem__(self, idx):
        if len(self._data) == 1:
            return MyTransform(self._data[0][idx], self.opts).random_transform()
        else:
            return tuple(MyTransform(data[idx], self.opts).random_transform() for data in self._data)

    def __len__(self):
        return self._length


if __name__ == "__main__":
    x = np.zeros(shape=(5, 2, 10, 30, 30))

    d = MyDataset(x)
    data_loader = DataLoader(d, batch_size=2)

    for b in data_loader:
        print(b[0].shape)
