import time
import torch
import numpy as np
from CycleGAN.options.train_options import TrainOptions
from CycleGAN.data.create_numpy_data_loader import get_numpy_loader, define_transformer
from CycleGAN.models import create_model
from CycleGAN.util.visualizer import Visualizer
from CycleGAN.util.load_npy import load_npy
import pylab as plt
from CycleGAN.util.norm_data import norm_01, norm


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # data_loader = CreateDataLoader(opt)
    A, B = load_npy(crop_size=opt.crop_size)
    if opt.train_prostateX_only:
        A = A[1402:]
    A = norm(A) if opt.norm_data else A

    data_loaderA = get_numpy_loader(A, opt=opt, transform=define_transformer(output_size=opt.input_size,
                                                                             translation=opt.translation))
    data_loaderB = get_numpy_loader(B, opt=opt, transform=define_transformer(output_size=opt.input_size,
                                                                             translation=opt.translation))
    dataset_size = len(data_loaderA.dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # for i, data in enumerate(dataset):
        for i, (dataA, dataB) in enumerate(zip(data_loaderA, data_loaderB)):
            iter_start_time = time.time()
            data = dict()
            data['A'], data['B'] = dataA[0].type(torch.FloatTensor), dataB[0].type(torch.FloatTensor)
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            # plt.imshow(model.fake_A[0, 0].detach().cpu().numpy(), cmap='gray')

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
