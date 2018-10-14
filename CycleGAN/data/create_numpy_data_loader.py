import torch
import torch.utils.data as utils
import numpy as np
from torch.utils.data.dataset import random_split, Subset
from torchvision.transforms import transforms
from CycleGAN.util.augmentations import *
from scipy import rot90
import numbers


class MyNumpyDataset(utils.Dataset):
    """Dataset wrapping Numpy.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *arr (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *arrs, transform=None):
        assert all(arrs[0].shape[0] == arr.shape[0] for arr in arrs)
        self.transform = transform
        self.arrs = arrs

    def __getitem__(self, index):
        if self.transform is not None:
            return tuple(self.transform(arr[index]) for arr in self.arrs)
        else:
            return tuple(arr[index] for arr in self.arrs)

    def __len__(self):
        return self.arrs[0].shape[0]


def define_transformer(output_size=180, translation=None):
    """"""
    # transform_dict = {'center_crop': CenterCrop(180), 'rot90': RandomRot90()}
    transformer = transforms.Compose([CenterCrop(output_size=output_size, translation=translation),
                                      # RandomRot90(),
                                      ])
    return transformer


def get_numpy_loader(*data, opt, transform=None, test_phase=False):
    """Test custom dataset"""
    dataset = MyNumpyDataset(*data, transform=transform)
    if test_phase:
        opt.train_size = 1  # run translation on the whole dataset
    if opt.train_size <= 1:
        train_size = int(len(dataset) * opt.train_size)
        test_size = len(dataset) - train_size
    else:
        train_size = opt.train_size
        test_size = len(dataset) - opt.train_size

    if opt.train_size == 1:
        train_dataset = dataset
    else:
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # test_idx = [0, 1, 3]
    # train_idx = list(set(range(10)) - set(test_idx))
    # train_dataset, test_dataset = Subset(dataset, train_idx), Subset(dataset, [0, 1, 3])

    train_dataloader = utils.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=not test_phase)
    return train_dataloader


if __name__ == "__main__":
    x = np.random.rand(10, 2, 160, 160)
    y = np.random.rand(10, 2, 160, 160)
    z = np.random.rand(10, 1)

    # train_dataloader = get_tensor_loader(x, y, z)
    train_dataloader = get_numpy_loader(x, y, transform=define_transformer())
    for batch in train_dataloader:
        print('abc')
        print(batch.__len__())
        print(batch[0].__len__())
        print(batch[0][0].shape)
        print(batch[1][0].shape)
        break
