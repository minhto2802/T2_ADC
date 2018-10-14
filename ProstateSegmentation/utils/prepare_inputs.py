import numpy as np
import pylab as plt
from augmentations import center_crop


def concat_inputs(exp_name='T2_ADC_run11', epoch=49, intermediate_dir='', which_set='training', output_size=180, input_size=240, isTraining=True):
    """Load and concatenate different MR modalities & groundtruth as different channels"""
    # intermediate_dir = '/histmatch_input';
    output_size = [output_size, output_size] if isinstance(output_size, int) else output_size
    dirs_fake_ADC = r'F:\Minh\projects\T2_ADC\CycleGAN\results\%s%s\test_%s_TrainingSet\fake_B.npy' % (exp_name, intermediate_dir, epoch)
    T2 = np.expand_dims(center_crop(output_size, np.load('F:\Minh\projects\ProstateX\data\T2_combined_%d_preproc_histmatch_PM12ONLY.npy' % input_size)), 1)
    ADC = np.load('%s' % dirs_fake_ADC)
    subIdx = np.load(r'F:\Minh\PROMISE2012\T2_ADC\im_matched\%s\npy\subIdx_%d.npy' % (which_set, input_size)).T[0]
    if isTraining:
        lab = np.expand_dims(center_crop(output_size, np.load(r'F:\Minh\PROMISE2012\T2_ADC\im_matched\%s\npy\LABs_%d.npy' % (which_set, input_size))), 1)
        return np.concatenate((T2, ADC, lab), axis=1), subIdx, isTraining, dirs_fake_ADC
    else:
        return np.concatenate((T2, ADC), axis=1), subIdx, isTraining, dirs_fake_ADC


def form_3d(x, subIdx, preset_size=2000, stride=6, zdim=16):
    """forming 3D volumes from 2D slices"""
    n = int(np.max(subIdx))
    x3d = np.zeros((preset_size, zdim) + x.shape[1:])
    subIdx3d = np.zeros(preset_size,)
    subIdx -= 1
    count = 0
    for i in range(n):
        print('%d ' % i)
        start_slice_idx = 0
        tmp = x[subIdx == i]
        while True:
            end_slice_idx = start_slice_idx + zdim
            if end_slice_idx > tmp.shape[0]:
                end_slice_idx = tmp.shape[0]
                start_slice_idx = end_slice_idx - zdim
            if get_number_of_slices_having_prostate(tmp[start_slice_idx: end_slice_idx, 2]) >= int(zdim / 2):
                x3d[count] = tmp[start_slice_idx: end_slice_idx]
                subIdx3d[count] = i
                count += 1
            start_slice_idx += stride
            if end_slice_idx == tmp.shape[0]:
                break
    return x3d[:count], subIdx3d[:count]


def get_number_of_slices_having_prostate(vol3d):
    """How many slices in a volume containing prostate?"""
    return np.sum(np.sum(vol3d, axis=(1, 2)) > 0)


def plot_x3d(x3d, vol_idx=32, idx=15, isTraining=True):
    """Visualize the one slice of a 3D volume"""
    x = x3d[vol_idx]
    plt.subplot(121), plt.imshow(x[idx, 0], cmap='gray', vmin=0, vmax=1)
    plt.contour(x[idx, 2], colors='y') if isTraining else None
    plt.subplot(122), plt.imshow(x[idx, 1], cmap='gray', vmin=0, vmax=1)
    plt.contour(x[idx, 2], colors='y') if isTraining else None


if __name__ == '__main__':
    to_plot = False
    x, subIdx, isTraining, dirs_out = concat_inputs()
    print(x.shape)
    if to_plot:
        idx = 44
        plt.subplot(121), plt.imshow(x[idx, 0], cmap='gray', vmin=0, vmax=1)
        plt.contour(x[idx, 2], colors='y') if isTraining else None
        plt.subplot(122), plt.imshow(x[idx, 1], cmap='gray', vmin=0, vmax=1)
        plt.contour(x[idx, 2], colors='y') if isTraining else None
        plt.show()
    x3d, subIdx3d = form_3d(x, subIdx)
    np.save(dirs_out.replace('fake_B', 'ps_input'), x3d)
    np.save(dirs_out.replace('fake_B', 'ps_input_idx'), subIdx3d)






