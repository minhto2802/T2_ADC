from matplotlib import pyplot as plt
from T2_ADC.utils.my_func import *
from datetime import datetime
import time

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import numpy as np
from mxboard import SummaryWriter
import logging


def train():
    """training"""
    image_pool = ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)

    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)

    # define a summary writer that logs data and flushes to the file every 5 seconds
    sw = SummaryWriter(logdir='%s' % dir_out_sw, flush_secs=5, verbose=False)
    global_step = 0

    for epoch in range(epochs):
        if epoch == 0:
            netG.hybridize()
            netD.hybridize()
        #     sw.add_graph(netG)
        #     sw.add_graph(netD)

        tic = time.time()
        btic = time.time()
        train_data.reset()
        val_data.reset()
        iter = 0
        for local_step, batch in enumerate(train_data):
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            tmp = mx.nd.concat(batch.data[0], batch.data[1], batch.data[2], dim=1)
            tmp = augmenter(tmp, patch_size=128, offset=offset, aug_type=1, aug_methods=aug_methods, random_crop=False)
            real_in = tmp[:, :1].as_in_context(ctx)
            real_out = tmp[:, 1:2].as_in_context(ctx)
            m = tmp[:, 2:3].as_in_context(ctx)  # mask

            fake_out = netG(real_in) * m

            # loss weight based on mask, applied on L1 loss
            if no_loss_weights:
                loss_weight = m
            else:
                loss_weight = m.asnumpy()
                loss_weight[loss_weight == 0] = .1
                loss_weight = mx.nd.array(loss_weight, ctx=m.context)

            fake_concat = image_pool.query(nd.concat(real_in, fake_out, dim=1))
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history images
                output = netD(fake_concat)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label, ], [output, ])

                # Train with real image
                real_concat = nd.concat(real_in, real_out, dim=1)
                output = netD(real_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD = (errD_real + errD_fake) * 0.5
                errD.backward()
                metric.update([real_label, ], [output, ])

            trainerD.step(batch.data[0].shape[0])

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_out = netG(real_in)
                fake_concat = nd.concat(real_in, fake_out, dim=1)
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = GAN_loss(output, real_label) + loss_2nd(real_out, fake_out, loss_weight) * lambda1
                errG.backward()

            trainerG.step(batch.data[0].shape[0])

            sw.add_scalar(tag='loss', value=('d_loss', errD.mean().asscalar()), global_step=global_step)
            sw.add_scalar(tag='loss', value=('g_loss', errG.mean().asscalar()), global_step=global_step)
            global_step += 1

            if epoch + local_step == 0:
                sw.add_graph((netG))
                img_in_list, img_out_list, m_val = val_data.next().data
                m_val = m_val.as_in_context(ctx)
                sw.add_image('first_minibatch_train_real', norm3(real_out))
                sw.add_image('first_minibatch_val_real', norm3(img_out_list.as_in_context(ctx)))
                netG.export('%snetG' % dir_out_checkpoints)
            if local_step == 0:
                # Log the first batch of images of each epoch (training)
                sw.add_image('first_minibatch_train_fake', norm3(fake_out * m) * m, epoch)
                sw.add_image('first_minibatch_val_fake',
                             norm3(netG(img_in_list.as_in_context(ctx)) * m_val) * m_val, epoch)
                             # norm3(netG(img_in_list.as_in_context(ctx)) * m_val.as_in_context(ctx)), epoch)

            if (iter + 1) % 10 == 0:
                name, acc = metric.get()

                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info(
                    'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                    % (nd.mean(errD).asscalar(),
                       nd.mean(errG).asscalar(), acc, iter, epoch))

            iter += 1
            btic = time.time()

        sw.add_scalar(tag='binary_training_acc', value=('acc', acc), global_step=epoch)

        name, acc = metric.get()
        metric.reset()

        fake_val = netG(val_data.data[0][1].as_in_context(ctx))
        loss_val = loss_2nd(val_data.data[1][1].as_in_context(ctx), fake_val, val_data.data[2][1].as_in_context(ctx)) * lambda1
        sw.add_scalar(tag='loss_val', value=('g_loss', loss_val.mean().asscalar()), global_step=epoch)

        if (epoch % check_point_interval == 0) | (epoch == epochs-1):
            netD.save_params('%snetD-%04d' % (dir_out_checkpoints, epoch))
            netG.save_params('%snetG-%04d' % (dir_out_checkpoints, epoch))

        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))

    sw.export_scalars('scalar_dict.json')
    sw.close()


if __name__ == "__main__":
    '''SET TRAINING PARAMETERS'''
    run_id = 50
    fold = 5
    gpu_id = 1

    no_mask = True
    prostate_segmentation = False
    no_loss_weights = True
    unmasked_T2 = True
    no_mask_at_all = True

    epochs = 1000
    batch_size = 20
    batch_size_val = 32
    check_point_interval = 20
    use_gpu = True
    ctx = mx.gpu(gpu_id) if use_gpu else mx.cpu()
    norm_data = True
    lr = 1e-3
    beta1 = .5
    lambda1 = 100
    optimizer = 'adam'

    aug_methods = {'flip': random_flip}

    offset = [-5, 5, -5, 5]  # for translation in augmentation

    pool_size = 50

    '''DOWNLOAD AND PREPROCESS DATASET'''
    dataset = 'AMC_GAN_T2_ADC'

    dir_in = 'F:\Minh\mxnet\projects\cancer_segmentation\inputs/'
    dir_out = r"F:/BACKUPS/%s/outputs/run%d/" % (dataset, run_id)
    dir_out_checkpoints = r"%s/checkpoints/" % dir_out
    dir_out_sw = r"%s/logs/" % dir_out

    input_file_suffix = ''

    for my_dir in [dir_out, dir_out_checkpoints, dir_out_sw]:
        make_dirs(my_dir)

    # Set loggers
    # logger = logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    log_file = 'log.txt'
    logging.basicConfig(filename='%s/%s' % (dir_out, log_file),
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # coloredlogs.install()

    # Log all the passed function parameters
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    logger.info('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        logger.info("    %s = %s" % (i, values[i]))
    logger.info('Output directory: %s' % dir_out)

    # load data
    folds = np.load(r'%sfolds.npy' % dir_in)
    caseID = np.load(r'%scaseID.npy' % dir_in)[0]
    im = transform_data(np.load('%sim%s.npy' % (dir_in, input_file_suffix)))
    mask = transform_data(np.load('%smask%s.npy' % (dir_in, input_file_suffix)))

    # remove cancer ROIs (unnecessary for this task)
    mask[mask == 2] = 1
    mask_ = np.zeros(mask.shape) + mask
    if prostate_segmentation:
        im[:, 1] = mask[:, 0]
    if no_mask_at_all:
        mask[mask == 0] = 1
    elif unmasked_T2:
        im[:, 1] = im[:, 1] * mask[:, 0]
    else:
        im = im * mask

    # loss_weights = mask
    # loss_weights[mask == 0] = .1  # small penalty for the loss in background

    if no_mask:
        mask[mask == 0] = 1

    # normalize data
    if norm_data:
        logger.info('Normalizing data across the whole dataset... ')  # per channel
        im[:, :1] = norm1(im[:, :1], mask=mask_)
        im[:, 1:2] = norm2(im[:, 1:2], mask=mask_)

    im = np.concatenate((im, mask), axis=1)

    # split dataset
    all_idx = np.arange(caseID.__len__())  # update all_idx
    if fold is not None:
        logger.info('Split dataset in to training and validation (fold %d)' % fold)
        # train_amount = int(round(train_percent * im.shape[0]))
        training_set = {}
        training_set['fold%d' % fold] = np.argwhere(folds[0] != fold).transpose().tolist()
        s = set(training_set['fold%d' % fold][0])
        train_idx = [i for i in all_idx if caseID[i] in s]
        val_idx = [i for i in all_idx if caseID[i] not in s]
        im_train = im[train_idx]
        im_val = im[val_idx]

    train_data = create_iterators(im_train, batch_size)
    val_data = create_iterators(im_val, batch_size_val)

    # preview_train_data()

    # Loss
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L1Loss()
    L2_loss = gluon.loss.L2Loss()
    SmoothL1_loss = SmoothL1Loss()
    loss_2nd = L1_loss

    netG, netD, trainerG, trainerD = set_network(ctx, lr, beta1, optimizer=optimizer)

    train()


def print_result():
    """display output"""
    num_image = 4
    img_in_list, img_out_list, m = val_data.next().data
    for i in range(num_image):
        img_in = nd.expand_dims(img_in_list[i], axis=0)
        plt.subplot(2, 4, i+1)
        visualize(img_in[0])
        img_out = netG(img_in.as_in_context(ctx)) * m
        plt.subplot(2, 4, i+5)
        visualize(img_out[0])
    plt.show()


# print_result()
