"""cancer classification (AMC dataset)"""
# import mxnet as mx
from mxnet import ndarray as nd, io, gpu, init
from resnet_init import Resnet
from T2_ADC.pix2pix_T2_ADC_split_train_val import split_train_val
from T2_ADC.pix2pix_T2_ADC_funcs import *
import time
import argparse
import numpy as np
from mxnet import gluon, autograd
from mxboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import logging
import inspect


class CancerClassification:
    """cancer classification"""
    def __init__(self):
        super(CancerClassification, self).__init__()
        """get options"""
        parser = argparse.ArgumentParser(description='Run CNN training - Cancer classification (AMC dataset).')
        parser.add_argument('--run_id', type=int, default=17)
        parser.add_argument('--dir_out', type=str, default=r'F:\BACKUPS\Pix2Pix_T22ADC_cancer_classification/')
        parser.add_argument('--log_name', type=str, default='logs')
        parser.add_argument('--fold', type=int, default=1)
        parser.add_argument('--split', type=int, default=0)
        parser.add_argument('--num_stage', type=int, default=4)
        parser.add_argument('--last_pool_kernel_size', type=int, default=4)
        parser.add_argument('--first_stride', type=int, default=1)
        parser.add_argument('--num_class', type=int, default=2)
        parser.add_argument('--growth_rate', type=int, default=4, help='DenseNet params')
        parser.add_argument('--input_option', type=int, default=2, help='0 for T2, , 1 for ADC, 2 for both T2 and ADC')
        parser.add_argument('--use_fake', type=bool, default=True)
        parser.add_argument('--norm_data', type=bool, default=True)
        parser.add_argument('--num_epochs', type=int, default=500)
        parser.add_argument('--resumed_epoch', type=int, default=0)
        parser.add_argument('--save_checkpoint_interval', type=int, default=10)
        parser.add_argument('--num_group', type=int, default=1, help='cardinality')
        parser.add_argument('--train_batch_size', type=int, default=128)
        parser.add_argument('--val_batch_size', type=int, default=64)
        parser.add_argument('--optimizer', type=str, default='sgd')
        parser.add_argument('--base_lr', type=int, default=1e-1)
        parser.add_argument('--lr_decay_interval', type=int, default=50)
        parser.add_argument('--lr_decay_rate', type=int, default=.9)
        parser.add_argument('--patch_size', type=int, default=64)
        parser.add_argument('--wd', type=int, default=1e-5, help='weight decay')
        parser.add_argument('--losses', default=['cross-entropy'], nargs='+', help='losses aimed to optimize')
        parser.add_argument('--network', type=str, default='Resnet')
        parser.add_argument('--units', type=int, default=[3, 4, 6, 3])
        parser.add_argument('--num_filters_list', type=int, default=[16, 64, 128, 256, 512])  # [16, 64, 128, 256, 512]
        parser.add_argument('--num_fpg', type=int, default=np.NaN, help='number of features in each group of grouped convolution')
        parser.add_argument('--gpu_id', type=int, nargs='+', default=1)
        self.args = parser.parse_args()
        self.networks = {'resnet': Resnet}
        self.losses = {'cross-entropy': gluon.loss.SoftmaxCrossEntropyLoss}


if __name__ == '__main__':
    CC = CancerClassification()
    opts = CC.args

    # assign and create directories
    dir_out = opts.dir_out + 'run%03d/' % opts.run_id
    dir_out_sw = dir_out + opts.log_name
    dir_out_checkpoints = dir_out + 'checkpoints/'
    for my_dir in [dir_out, dir_out_checkpoints, dir_out_sw]:
        make_dirs(my_dir)

    # Set loggers
    logging.basicConfig(filename='%s/%s' % (dir_out, 'logs.txt'),
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Log all the passed function parameters
    frame = inspect.getmembers(opts)
    for fr in frame:
        if fr[0] is '__dict__':
            for i in fr[1]:
                logger.info("    %s = %s" % (i, fr[1][i]))

    # define logging on mxboard
    sw = SummaryWriter(dir_out_sw, max_queue=20, flush_secs=5, filename_suffix='cc_', verbose=False)

    # context
    opts.gpu_id = [opts.gpu_id] if not isinstance(opts.gpu_id, list) else opts.gpu_id
    ctx = gpu(opts.gpu_id[0])

    # define network
    network = CC.networks[opts.network.lower()]
    net = network(units=opts.units, num_filters_list=opts.num_filters_list, num_stage=opts.num_stage,
                  num_class=opts.num_class, first_strides=opts.first_stride, last_pool_kernel_size=opts.last_pool_kernel_size)

    # load dataset
    im_train, im_val, c_labels_train, c_labels_val, mask_train, mask_val = split_train_val(opts.fold, opts.use_fake, opts.norm_data)

    # specify the used channels in input images
    if opts.input_option != 2:
        im_train, im_val = im_train[:, opts.input_option, None],  im_val[:, opts.input_option, None]

    # parameters initializer
    if opts.resumed_epoch == 0:
        net.collect_params().initialize(init.Xavier(magnitude=2), ctx=ctx)
    else:
        net.load_params('%snet-%04d' % (dir_out_checkpoints, opts.resumed_epoch))

    # define trainer
    trainer = gluon.Trainer(net.collect_params(), optimizer=opts.optimizer,
                            optimizer_params={'learning_rate': opts.base_lr, 'wd': opts.wd})

    # define data iteration
    train_iter = io.NDArrayIter({'data': im_train, 'mask': mask_train}, {'label': c_labels_train},
                                batch_size=opts.train_batch_size, shuffle=True)
    val_iter = io.NDArrayIter({'data': im_val, 'mask': mask_val}, {'label': c_labels_val},
                              batch_size=opts.val_batch_size)

    # losses
    losses = {}
    for l in opts.losses:
        losses[l] = CC.losses[l]()

    # validation data to GPU
    im_val = nd.array(im_val, ctx=ctx)
    label_val = nd.array(c_labels_val, ctx=ctx)

    # metrics
    def get_accuracy(p, lab):
        """compute classification accuracy"""
        return (p == lab).sum() / p.shape[0]

    # hybridize network for faster performance
    x = nd.random_normal(1, .02, shape=(opts.train_batch_size, im_train.shape[1], opts.patch_size, opts. patch_size), ctx=ctx)
    _ = net(x)
    net.hybridize()

    # main training loop
    global_step = 0
    max_val_acc = 0
    for e in range(opts.num_epochs):
        start_e = time.time()
        train_iter.reset()
        val_iter.reset()
        acc_accumulate = []
        acc_accumulate_val = []
        loss_accumulate = 0
        loss_accumulate_val = 0
        pred_val_accumulate = []
        label_val_accumulate = []
        # loop through all of training data
        for i, batch in enumerate(train_iter):
            start_b = time.time()
            data_ = batch.data[0]
            mask_ = batch.data[1]
            label = batch.label[0].as_in_context(ctx)

            # randomly crop patches
            aum = Augmenter(aug_list=['deform', 'hist_norm'], exclude=True)
            data = np.zeros((data_.shape[:2] + (opts.patch_size, opts.patch_size)))
            mask = np.zeros((mask_.shape[:2] + (opts.patch_size, opts.patch_size)))
            for im_idx in range(data_.shape[0]):
                data[im_idx], mask[im_idx] = crop_cancer_ROIs(data_[im_idx], mask_[im_idx], label[im_idx])
                data[im_idx] = aum.forward(data[im_idx])
                # visualize(im[im_idx, 1], mask[im_idx, 0])
            data = nd.array(data, ctx=ctx)

            loss = 0
            with autograd.record():
                pred = net(data)
                for l in losses.keys():
                    loss = loss + losses[l](pred, label)
            loss.backward()

            acc_accumulate.append(get_accuracy(nd.argmax(pred, axis=1).asnumpy(), label.asnumpy()))
            loss_accumulate = loss_accumulate + nd.sum(loss).asscalar()
            trainer.step(data.shape[0])

            if i % 5 == 0:
                logger.info('Epoch %d/%d,  batch %d: accuracy = %.4f,  speed: %.2f Hz' %
                            (e, opts.num_epochs, i, acc_accumulate[-1], data.shape[0]/(time.time() - start_b)))

            global_step += 1
            if (global_step % opts.lr_decay_interval == 0) & (global_step != 0):
                trainer.set_learning_rate(trainer.learning_rate * opts.lr_decay_rate)

        # loop through all of validation data
        for i, batch in enumerate(val_iter):
            data_ = batch.data[0]
            mask_ = batch.data[1]
            label = batch.label[0].as_in_context(ctx)

            # randomly crop patches
            data = np.zeros((data_.shape[:2] + (opts.patch_size, opts.patch_size)))
            mask = np.zeros((mask_.shape[:2] + (opts.patch_size, opts.patch_size)))
            for im_idx in range(data_.shape[0]):
                data[im_idx], mask[im_idx] = crop_cancer_ROIs(data_[im_idx], mask_[im_idx], label[im_idx])
                # visualize(im[im_idx, 1], mask[im_idx, 0])
            data = nd.array(data, ctx=ctx)

            pred_val = net(data)
            loss = 0
            for l in losses.keys():
                loss = loss + losses[l](pred_val, label)
            acc_accumulate_val.append(get_accuracy(nd.argmax(pred_val, axis=1).asnumpy(), label.asnumpy()))
            loss_accumulate_val = loss_accumulate_val + nd.sum(loss).asscalar()

            pred_val_accumulate.append(nd.argmax(pred_val, axis=1))
            label_val_accumulate.append(label)

            # get indexes of wrong predictions
            if False:
                u = list(pred_val.argmax_channel().asnumpy())  # prediction
                idx = [i for i in range(u.__len__()) if u[i] != label_val[i]]  # index of prediction == 0

        # compute the confusion matrix on validation set
        cm = confusion_matrix(nd.concat(*pred_val_accumulate, dim=0).asnumpy(),
                              nd.concat(*label_val_accumulate, dim=0).asnumpy())
        logger.info('[%d %d] [%d %d]' % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))

        logger.info('Epoch %d/%d: train [acc, loss] = [%.4f, %.4f],  '
                    'val [acc, loss] = [%.4f, %.4f],  LR = %.5f, GS = %d' %
                    (e, opts.num_epochs, np.array(acc_accumulate).mean(), loss_accumulate/im_train.shape[0],
                     np.array(acc_accumulate_val).mean(), loss_accumulate_val/im_val.shape[0],
                     trainer.learning_rate, global_step))

        sw.add_scalar(tag='acc', value=('training', np.array(acc_accumulate).mean()), global_step=e)
        sw.add_scalar(tag='acc', value=('validation', np.array(acc_accumulate_val).mean()), global_step=e)

        sw.add_scalar(tag='loss', value=('training', loss_accumulate/im_train.shape[0]), global_step=e)
        sw.add_scalar(tag='loss', value=('validation', loss_accumulate_val/im_val.shape[0]), global_step=e)

        if np.array(acc_accumulate_val).mean() > max_val_acc:
            max_val_acc = np.array(acc_accumulate_val).mean()
        logger.info('Max validation accuracy %.4f' % max_val_acc)

        if e % opts.save_checkpoint_interval == 0:
            net.save_params(r'%snet-%04d' % (dir_out_checkpoints, e))
            logger.info('Save parameters')

        logger.info('Time per epoch = %.2f s' % (time.time() - start_e))
        print()


