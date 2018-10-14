import numpy as np
from glob import glob
import pylab as plt
from utils.utils import dice_wp, post_proc

if __name__ == "__main__":
    n_rows = 4
    dataset = 'PROMISE'
    run_id = 89
    epoch = 44
    dir_out = "F:\BACKUPS\%s\outputs/run%d/" % (dataset, run_id)
    dir_out_vols = '{:s}/vols_{:04d}/'.format(dir_out, epoch)
    dice_list = []
    show_fig = True
    to_post_proc = True
    for i in range(50):
        try:
            fname_pred = glob("{:s}vol_pred_{:03d}_*".format(dir_out_vols, i))[0]
        except:
            break
        fname_gt = glob("{:s}vol_gt_{:03d}_*".format(dir_out_vols, i))[0]
        fname_im = glob("{:s}vol_im_{:03d}_*".format(dir_out_vols, i))[0]
        vol_pred = np.load(fname_pred)
        vol_gt = np.load(fname_gt)
        vol_im = np.load(fname_im)

        # post-processing
        if to_post_proc:
            vol_pred = post_proc(vol_pred)

        dice = dice_wp(vol_pred[np.newaxis], vol_gt[np.newaxis]).asscalar()
        dice_list.append(dice)
        print(dice)
        dice_per_slice = dice_wp(vol_pred, vol_gt).asnumpy()
        plt.figure(i)

        if show_fig:
            for j in range(vol_pred.shape[0]):
                plt.subplot(n_rows, np.ceil(vol_pred.shape[0]/n_rows), j+1)
                plt.imshow(vol_im[1, j], cmap='gray', vmin=0, vmax=1)
                plt.contour(vol_gt[j], colors='y', linewidths=.5)
                plt.contour(vol_pred[j], colors='r', linewidths=.5)
                plt.axis('off')
                plt.title('{:.2f}'.format(dice_per_slice[j]))
            plt.show()

    print('Mean dice: {:.3f}'.format(np.array(dice_list).mean()))

