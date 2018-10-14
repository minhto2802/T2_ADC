import numpy as np
import mxnet as mx
import cv2
import scipy.ndimage
import inspect


def transform_image(im, lab, r_int):
    import cv2

    def augment_brightness_camera_images(image):
        image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()
        # print(random_bright)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
        return image1

    def transform_image_(img, ang_range, shear_range, trans_range, brightness=0):
        '''
        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over.

        A Random uniform distribution is used to generate different parameters for transformation

        '''
        # Rotation

        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols = img[:, :, 30, 0, 0].shape
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

        # Brightness
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

        shear_M = cv2.getAffineTransform(pts1, pts2)
        for i in np.arange(img.shape[-1]):
            for j in np.arange(img.shape[-2]):
                img[:, :, :, j, i] = cv2.warpAffine(img[:, :, :, j, i], Rot_M, (cols, rows))
                img[:, :, :, j, i] = cv2.warpAffine(img[:, :, :, j, i], Trans_M, (cols, rows))
                img[:, :, :, j, i] = cv2.warpAffine(img[:, :, :, j, i], shear_M, (cols, rows))

        if brightness == 1:
            img = augment_brightness_camera_images(img)

        return img

    im = np.transpose(im.asnumpy(), tuple(np.arange(-1, -len(im.shape)-1, -1)))
    lab = np.reshape(np.transpose(lab.asnumpy(), tuple(np.arange(-1, -len(lab.shape) - 1, -1))),
                     tuple((im.shape[0], im.shape[1], im.shape[2], 1, im.shape[4])))
    image = transform_image_(np.concatenate((im, lab), axis=3), 30, 5, 0, brightness=0)
    image = np.transpose(image[:, :, 0:im.shape[2], :, :], tuple(np.arange(-1, -len(im.shape) - 1, -1)))
    im = image[:, 0:image.shape[1]-1]
    lab = image[:, image.shape[1]-1]
    # tmp = (im.shape[0], im.shape[2:])
    # lab = np.reshape(np.transpose(image[:, :, im.shape[2]:, :, :], tuple(np.arange(-1, -len(lab.shape) - 1, -1))), tuple([tmp[0]] + list(tmp[1])))
    return mx.nd.array(im), mx.nd.array(lab)


def random_rot(im, r_int):
    ctx = im.context
    im = scipy.ndimage.interpolation.rotate(im.asnumpy(), r_int, axes=(-2, -1), reshape=False, mode='reflect')
    return im


def random_rot90(im, r_int):
    ctx = im.context
    k = np.random.randint(1, 4)
    im = np.rot90(im.asnumpy(), k=k, axes=(-2, -1))
    return mx.nd.array(im, ctx=ctx)


def add_noise(im, r_int):
    mu = (np.mean(im.asnumpy(), axis=(-1, -2, -3)))/4
    sigma = np.sqrt(np.std(im.asnumpy(), axis=(-1, -2, -3)))
    im = im + mx.nd.sample_normal(mu=mx.nd.array(mu * np.random.rand()), sigma=mx.nd.array(sigma * np.random.rand()),
                                  shape=im.shape[1:])
    return im


def random_flip(im1, angle=0):
    ctx = im1.context
    ax = np.random.randint(-1, 0)
    im1 = np.flip(im1.asnumpy(), ax)
    return mx.nd.array(im1, ctx=ctx)


def translate_image(im1, angle=0):
    ctx = im1.context
    shift_y = np.random.randint(-2, 2)
    shift_x = np.random.randint(-2, 2)
    im1 = np.roll(im1.asnumpy(), (shift_y, shift_x), axis=(-2, -1))
    return mx.nd.array(im1, ctx=ctx)


def translate_by_slide(im1=None, angle=0):
    x = im1.asnumpy()
    for i in np.arange(x.shape[0]):
        for j in np.arange(x.shape[2]):
            shift_y = np.random.randint(-1, 1)
            shift_x = np.random.randint(-1, 1)
            x[i, :, j] = np.roll(x[i, :, j], (shift_y, shift_x), axis=(-2, 1))
    return mx.nd.array(x, ctx=im1.context)


def scale_image(im1=None, angle=0):
    sh_im_org = im1.shape
    c = int(im1.shape[-1]/2)
    max_zooming_percent = .05
    scale_factor = max_zooming_percent * np.random.rand()
    new_s = int(im1.shape[-1] * (1 - scale_factor)/2)
    im1_cr = im1[:, :, c-new_s: c+new_s, c-new_s: c+new_s].asnumpy()
    im1_s = np.zeros(sh_im_org)
    for i in np.arange(im1.shape[0]):
        for j in np.arange(im1.shape[1]):
            im1_s[i, j] = cv2.resize(im1_cr[i, 0, j], (sh_im_org[-2], sh_im_org[-1]))
    return mx.nd.array(im1_s)
