import numbers
from scipy import rot90
import numpy as np
import torch


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, output_size, translation=None):
        if isinstance(output_size, numbers.Number):
            self.output_size = (int(output_size), int(output_size))
        else:
            self.output_size = output_size
        self.translation = translation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return center_crop(self.output_size, img, self.translation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.output_size)


class RandomRot90(object):
    """Crops the given PIL Image at the center."""
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        self.k = np.random.randint(1, 4)
        return random_rot_90(self.k, img)


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


def random_rot_90(k, img):
    """main function"""
    if np.random.rand() > .5:
        print(k)
        return np.ascontiguousarray(rot90(img, k, axes=(-2, -1)))
    else:
        return img
