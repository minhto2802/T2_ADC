import os
import torch
import numpy as np
from CycleGAN.options.test_options import TestOptions
from CycleGAN.data import CreateDataLoader
from CycleGAN.models import create_model
from CycleGAN.util.visualizer import save_images
from CycleGAN.util import html
from CycleGAN.util.load_npy import load_npy
from CycleGAN.data.create_numpy_data_loader import get_numpy_loader, define_transformer
from os import getcwd
from os.path import dirname


if __name__ == '__main__':
    opt = TestOptions().parse()
    # opt.filename = 'F:/Minh/PROMISE2012/T2_ADC/im_matched/test/npy/IMs_240_01_perSub_histmatch.npy'
    # opt.filename = r'F:\Minh\projects\ProstateX\data/T2_combined_240_preproc_histmatch_PM12ONLY.npy'
    opt.filename = r'F:\Minh\projects\DECATHLON\data_merged\imagesTr/full_cropped_240_01_perSub_hist_norm_2d.npy'
    # opt.filename = r'F:\Minh\projects\DECATHLON\data_merged\imagesTr/full_cropped_240_01_perSub_2d.npy'
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    A = load_npy(crop_size=opt.crop_size, test_phase=True,
                 filename=opt.filename
                 )
    B = torch.empty(A.shape[0], A.shape[1], opt.input_size, opt.input_size)
    data_loaderA = get_numpy_loader(A, opt=opt, transform=define_transformer(output_size=opt.input_size), test_phase=True)

    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, dataA in enumerate(data_loaderA):
        data = dict()
        data['A'] = dataA[0].type(torch.FloatTensor)
        data['A_paths'] = ['./s%04d' % i]
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        try:
            B[i:i+1] = model.fake_B
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        except:
            break
    np.save('{}/fake_B_.npy'.format(web_dir), B.numpy())
    webpage.save()

