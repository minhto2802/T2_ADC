"""Training the network for prostate segmentation"""
from mxnet import gluon, io, gpu, initializer, autograd, nd
from gluoncv import utils as gutils
from glob import glob
import pylab as plt
from utils.custom_dataset import MyDataset
from utils.init_params import InitParams
from utils.learning_rate_schedulers import OneCycleSchedule, CyclicalSchedule, TriangularSchedule
from scipy import ndimage

from networks import dmnet_init as net_init1
from networks import dmnet_multi_inputs_init as net_init2
from mxnet.gluon.data import DataLoader

from utils.utils import *
import numpy as np
from numpy import matlib
import argparse
import os
import logging
import time
import inspect
import pickle


def get_args():
    """get commandline parameters"""
    parser = argparse.ArgumentParser(description='Run CNN training on prostate segmentation.')
    parser.add_argument('--run_id', type=int, help='Training sessions ID')
    parser.add_argument('--dir_in', type=str, help='Path to input file')
    parser.add_argument('--dir_out', type=str, help='Path to save training session')
    parser.add_argument('--dir_retrieved_file', type=str,
                        help='Specify this in case that we want to reuse parameters from other training sessions, '
                             'Path to the training instance')
    parser.add_argument('--retrieved_params', type=str,
                        help='Name of parameters we want to retrieve, e.g. "base_lr, val_amount"')
    parser.add_argument('--split', type=int, help='Split index')
    parser.add_argument('--resumed_epoch', type=int, help='Epoch wished to resume')
    parser.add_argument('--train_batch_size', type=int, help='Size of each mini-batch in training')
    parser.add_argument('--val_batch_size', type=int, help='Size of each mini-batch in validation')
    parser.add_argument('--train_amount', type=int, help='Size of training set')
    parser.add_argument('--val_amount', type=int, help='Size of validation set')
    parser.add_argument('--steps', type=int, help='Total training steps, default: 10000')
    parser.add_argument('--gpu_id', type=int, help='GPUs used for training, e.g. 0, '
                                                   'leave empty to use CPUs, default: 0')
    parser.add_argument('--log_file', type=str, help='Name of the log file, default "log.txt"')
    parser.add_argument('--training_data_name', type=str,
                        help='Name of the training image file (include both training and validation set')
    parser.add_argument('--training_gt_name', type=str,
                        help='Name of the training ground truth file (include both training and validation set')
    parser.add_argument('--to_review_network', type=bool,
                        help='Get some information about the network, '
                             'including detailed architecture, number of parameters')
    parser.add_argument('--optimizer', type=str, help='Name of the optimizer for optimizing network, default: "adam"')
    parser.add_argument('--loss_term', type=str, help='Name of the loss term we want to minimize, default: CrossEntropy')
    parser.add_argument('--base_lr', type=float, help='Initial learning rate, default: 1e-3 or 0.001')
    parser.add_argument('--wd', type=float, help='Weight decay, default: 5e-4 or 0.0005')
    parser.add_argument('--norm_data', type=int, help='whether normalize to 0mean and 1std (set 1) or not (set 0)')
    parser.add_argument('--seed', type=int,
                        help='A number specifying which fixed seed is used for MXNet, Numpy and Python')
    parser.add_argument('--log_interval', type=int, help='Logging interval, default: 5')
    parser.add_argument('--save_interval', type=int, help='Number of epochs between each checkpoints, default: 1')
    parser.add_argument('--val_interval', type=int, help='Number of epochs between each validation, default: 1')
    parser.add_argument('--prefix', type=str, help='Checkpoint name prefix, default: "dmnet"')
    parser.add_argument('--max_angle', type=int, help='range of possible rotating angles during augmentation')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--transform_list_idx', type=list, help='specify the indexes of transformations')
    parser.add_argument('--label_in_last_channel', type=bool, help='whether last label is concatenated with input images')
    parser.add_argument('--activation', type=str, help='whether last label is concatenated with input images')
    parser.add_argument('--not_use_ADC', action='store_true', help='whether last label is concatenated with input images')
    args = parser.parse_args()
    # args.use_ADC = not args.not_use_ADC
    return args


class Training(InitParams):
    """Create a training sessions"""
    def __init__(self, args):
        super(Training, self).__init__()
        self.get_args(args)
        self.set_context()

    def get_args(self, args):
        """Assign arguments from command lines to the Training class"""
        for key, value in zip(args.__dict__.keys(), args.__dict__.values()):
            if value is not None:
                self.__setattr__(key, value)

    def fix_seed(self):
        """Fix seed for mxnet, numpy and python builtin random generator"""
        gutils.random.seed(self.seed)

    def list_parameters(self, logger):
        """List and log all current parameters"""
        params = inspect.getmembers(self)
        logger.info("List all parameters...")
        for param in params:
            if not param[0].__contains__('__') and not callable(param[1]):
                if param[1] is not None:
                    logger.info("   {}: {}".format(param[0], param[1]))

    def prepare_dirs(self):
        """specifying input and output directories"""
        if not os.path.isdir(self.dir_out):
            os.makedirs(self.dir_out)
        if not os.path.isdir(self.dir_fig):
            os.makedirs(self.dir_fig)
        if self.val_only:
            if not os.path.isdir(self.dir_out_vols):
                os.makedirs(self.dir_out_vols)

    def get_coor_maps(self):
        my = matlib.repmat(np.linspace(0, 1, self.crop_size), self.crop_size, 1)
        mx = my.T
        self.coor_maps = np.tile(np.tile(np.concatenate((mx[np.newaxis], my[np.newaxis]), axis=0)[:, np.newaxis],
                                 (1, 41, 1, 1))[np.newaxis], (self.train_batch_size, 1, 1, 1, 1))

    def load_data(self):
        """load inputs"""
        # a = np.load(r'F:\Minh\projects\T2_ADC\CycleGAN\results\T2_ADC_run11\test_49_TrainingSet/ps_input.npy')
        # b = np.load(r'F:\Minh\projects\T2_ADC\CycleGAN\results\T2_ADC_run11\test_49_TrainingSet/ps_input_idx.npy')
        # self.im = np.load("%s%s.npy" % (self.dir_in, self.training_data_name))
        # self.lab = np.load("%s%s.npy" % (self.dir_in, self.training_gt_name))
        # im = np.load(r'%s/T2_ADC%s.npy' % (self.dir_in, self.file_suffix))
        file_suffix = '_normed' if self.already_normed else ''
        im = np.load(r'%s/IMs_LABs_%d%s.npy' % (self.dir_in_tmp, self.full_size, file_suffix))
        if self.use_ADC_only:
            im = im[:, -1:]
        elif not self.use_ADC:
            im = im[:, :-1]
        if (self.zcoor_maps is not None) and (not self.zcoor_maps):
            im = im[:, 1:]
        if self.norm_data:
            im = norm_0mean(im)
        # lab = np.load(r'%s/T2_ADC_lab%s.npy' % (self.dir_in, self.file_suffix))
        # self.im = np.concatenate((im, lab), axis=1).astype('float32')
        self.im = im
        self.idx = np.load(r'%s/T2_ADC_idx%s.npy' % (self.dir_in, self.file_suffix)) - 1
        self.sl_idx = np.load(r'%s/T2_ADC_slice_idx%s.npy' % (self.dir_in, self.file_suffix)).astype('int') - 1

    def split_data(self):
        """split dataset into training & validation"""
        # splits = np.loadtxt('%s/splits.txt' % self.dir_in, 'int')
        # self.train_idx = splits[self.split][:self.train_amount]
        # self.val_idx = splits[self.split][self.train_amount:(self.train_amount + self.val_amount)]
        splits = np.load('%s/T2_ADC_val_idx%s.npy' % (self.dir_in, self.file_suffix))[self.split]
        self.train_idx = np.where(splits == 0)[0]
        self.val_idx = np.where(splits == 1)[0]
        self.val_idx_unique = (np.unique(self.idx[self.val_idx].T[0])).astype('int')

    def transform_data(self):
        """transpose images to NCDHW (im) and NDHW (lab)"""
        self.im = np.transpose(self.im, (4, 3, 2, 0, 1))
        self.lab = np.transpose(self.lab, (3, 2, 0, 1))

    def set_logger(self):
        """Create logger for logging training process"""
        logging.basicConfig(filename='%s/%s' % (self.dir_out, self.log_file),
                            format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        return logger

    def exclude_samples(self):
        """Throw away samples with number of slices containing GT less than 2/3 of the number slices"""
        exclude_list = []
        thr = self.im.shape[-3] * (1/5)
        for i in range(self.train_idx.__len__()):
            n = (self.im[self.train_idx[i], -1].sum(axis=(-1, -2)) > 0).sum()
            if n < thr:
                exclude_list.append(i)
        self.train_idx = np.array([self.train_idx[i] for i in range(self.train_idx.__len__()) if i not in exclude_list])

    def set_iter_(self):
        """Prepare data iterator for training and validation"""
        self.train_iter = io.NDArrayIter(self.im[self.train_idx],
                                         self.lab[self.train_idx],
                                         batch_size=self.train_batch_size, shuffle=True)
        self.val_iter = io.NDArrayIter(self.im[self.val_idx],
                                       self.lab[self.val_idx],
                                       batch_size=self.val_batch_size)

    def set_iter(self):
        self.is_val = False
        self.training_set = MyDataset(self, self.im[self.train_idx])
        self.is_val = True
        self.val_set = MyDataset(self, self.im[self.val_idx])
        self.train_iter = DataLoader(self.training_set, batch_size=self.train_batch_size, shuffle=True)
        self.val_iter = DataLoader(self.val_set, batch_size=self.val_batch_size)

    def set_context(self):
        """set up context (GPUs or CPUs)"""
        # self.ctx = [gpu(int(i)) for i in self.gpus.split(',') if i.strip()][0]
        # self.ctx = self.ctx if self.ctx else [cpu()]
        self.ctx = gpu(self.gpu_id)

    def set_trainer(self):
        """Set up training policy"""
        if self.lr_scheduler is 'cycle':
            schedule = CyclicalSchedule(TriangularSchedule, min_lr=self.min_lr, max_lr=self.max_lr, cycle_length=200)
        elif self.lr_scheduler is 'one_cycle':
            schedule = OneCycleSchedule(min_lr=self.min_lr, max_lr=self.max_lr, cycle_length=450, cooldown_length=750, finish_lr=1e-4)
        else:
            schedule = None
        self.optimizer_params = {'learning_rate': self.base_lr,
                             'lr_scheduler': schedule,
                             'wd': self.wd,
                             'clip_gradient': None,
                             # 'rescale_grad': 1.0 / len(self.gpu_id) if len(self.gpu_id) > 0 else 1.0}
                             }
        self.Trainer = gluon.Trainer(self.model.net.collect_params(), optimizer=self.optimizer,
                                     optimizer_params=self.optimizer_params)
        # if self.resumed_epoch > -1:
        #     self.Trainer.load_states('{:s}_{:04d}.states'.format(self.dir_out + self.prefix, self.resumed_epoch))

    def set_loss(self):
        """Define loss terms used for network optimization """
        loss_dict = {'CrossEntropy': CELoss(axis=1),
                     'DiceLoss': DiceLoss()}
        self.loss = loss_dict[self.loss_term]

    def save_instance(self):
        """save the Training instance as self.name.txt"""
        with open(self.dir_out + "Training.file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def retrieve_params(self, dir_retrieved_file, param_list):
        """Retrieve the Training parameters from other Training session
        retrieved_file: path to the file of the Training instance, which was saved by save_instance
        param_list: str or list, includes the name of parameters we want to retrieve
        """
        with open(dir_retrieved_file + "Training.file", "rb") as f:
            dump = pickle.load(f)
        for param in param_list:
            self.__setattr__(param, dump.__getattr(param))

    def get_dice_wp(self, preds):
        """Compute dice per subject"""
        dice_val = []
        for jji, jj in enumerate(self.val_idx_unique):
            gt = self.im[self.val_idx, -1][self.idx[self.val_idx].T[0] == jj]
            im = self.im[self.val_idx, -3:-1][self.idx[self.val_idx].T[0] == jj]
            pred = preds[self.idx[self.val_idx].T[0] == jj]
            sl_idx = self.sl_idx[int(jj)][self.sl_idx[int(jj)] > -1]

            vol_im = np.zeros(shape=(2, sl_idx[-1] + 1, gt.shape[-2], gt.shape[-1]))
            vol_gt = np.zeros(shape=(sl_idx[-1] + 1, gt.shape[-2], gt.shape[-1]))
            vol_pred = np.zeros(shape=(2, sl_idx[-1] + 1, gt.shape[-2], gt.shape[-1]))
            vol_count = np.ones(vol_pred.shape)

            for kki, kk in enumerate(range(0, sl_idx.__len__(), 2)):
                vol_im[:, sl_idx[kk]: sl_idx[kk + 1] + 1] = im[kki][:, : (sl_idx[kk+1] - sl_idx[kk] + 1)]
                vol_gt[sl_idx[kk]: sl_idx[kk + 1] + 1] = gt[kki][: (sl_idx[kk+1] - sl_idx[kk] + 1)]
                vol_pred[:, sl_idx[kk]: sl_idx[kk + 1] + 1] += pred[kki][:, : (sl_idx[kk+1] - sl_idx[kk] + 1)]
                vol_count[:, sl_idx[kk]: sl_idx[kk + 1] + 1] += 1
            vol_pred /= vol_count
            vol_pred = post_proc(vol_pred.argmax(axis=0))
            dice_val.append(dice_wp(vol_pred[np.newaxis], vol_gt[np.newaxis]).expand_dims(0))
            if self.val_only:
                np.save("{:s}vol_pred_{:03d}_{:03d}".format(self.dir_out_vols, jji, jj), vol_pred)
                np.save("{:s}vol_gt_{:03d}_{:03d}".format(self.dir_out_vols, jji, jj), vol_gt)
                np.save("{:s}vol_im_{:03d}_{:03d}".format(self.dir_out_vols, jji, jj), vol_im)
        return dice_val

    def validate(self, epoch):
        preds = []
        dice_val = []
        loss_val = []
        if self.val_interval and ((epoch + 1) % self.val_interval == 0) and (epoch >= self.start_validation):
            if self.use_ADC:
                channel_idx = -2
            else:
                channel_idx = -1
            self.is_val = True
            for (i, idx) in enumerate(self.val_iter._batch_sampler):
                batch = nd.array(self.val_set[idx], ctx=self.ctx)
                if Training.use_ADC and Training.use_multi_branches:
                    x1 = batch[:-1, -2]
                    x2 = batch[:-1, [0, 1, -2]]
                else:
                    x = batch[:-1, :-1]
                gt = batch[:-1, -1]
                tmp = batch[-1, -1, 0].asnumpy()
                sl_idx = np.reshape(tmp[tmp > -999], (x.shape[0], -1)).astype('int')
                count_ = np.zeros((sl_idx.max() + 1))
                pred = np.zeros((2, sl_idx.max() + 1, x.shape[-2], x.shape[-1]))
                gt_wp = np.zeros((1, sl_idx.max() + 1, x.shape[-2], x.shape[-1]))
                x_wp = np.zeros((2, sl_idx.max() + 1, x.shape[-2], x.shape[-1]))
                tt = np.zeros(sl_idx.max() + 1)
                for ii in range(x.shape[0]):
                    if Training.use_ADC and Training.use_multi_branches:
                        if np.all((-100 < sl_idx[ii]) & (sl_idx[ii] <= 0)):
                            sl_idx[ii] = np.abs(sl_idx[ii])
                            pred[:, sl_idx[ii]] += np.flip(self.model.net(x1[ii:ii + 1], x2[ii:ii + 1])[0].asnumpy(),
                                                           axis=-1)
                        elif np.all((-200 < sl_idx[ii]) & (sl_idx[ii] <= -100)):
                            sl_idx[ii] = np.abs(sl_idx[ii] + 100)
                            pred[:, sl_idx[ii]] += np.flip(self.model.net(x1[ii:ii + 1], x2[ii:ii + 1])[0].asnumpy(),
                                                           axis=-2)
                            # pred_tmp = self.model.net(x[ii:ii + 1])[0].asnumpy()
                            # pred[:, sl_idx[ii]] += ndimage.rotate(pred_tmp, angle=+5, axes=[-2, -1], reshape=False)
                        elif np.all(sl_idx[ii] >= 0):
                            pred[:, sl_idx[ii]] += self.model.net(x1[ii:ii + 1], x2[ii:ii + 1])[0].asnumpy()
                            gt_wp[:, sl_idx[ii]] = gt[ii].asnumpy()
                            x_wp[:, sl_idx[ii]] = x[ii, channel_idx].asnumpy()
                    else:
                        if np.all((-100 < sl_idx[ii]) & (sl_idx[ii] <= 0)):
                            sl_idx[ii] = np.abs(sl_idx[ii])
                            pred[:, sl_idx[ii]] += np.flip(self.model.net(x[ii:ii + 1])[0].asnumpy(), axis=-1)
                        elif np.all((-200 < sl_idx[ii]) & (sl_idx[ii] <= -100)):
                            sl_idx[ii] = np.abs(sl_idx[ii] + 100)
                            pred[:, sl_idx[ii]] += np.flip(self.model.net(x[ii:ii + 1])[0].asnumpy(), axis=-2)
                            # pred_tmp = self.model.net(x[ii:ii + 1])[0].asnumpy()
                            # pred[:, sl_idx[ii]] += ndimage.rotate(pred_tmp, angle=+5, axes=[-2, -1], reshape=False)
                        elif np.all(sl_idx[ii] >= 0):
                            pred[:, sl_idx[ii]] += self.model.net(x[ii:ii + 1])[0].asnumpy()
                            gt_wp[:, sl_idx[ii]] = gt[ii].asnumpy()
                            x_wp[:, sl_idx[ii]] = x[ii, channel_idx].asnumpy()
                    tt[sl_idx[ii]] += 1
                    count_[sl_idx[ii]] += 1
                # pred /= count_[np.newaxis, :, np.newaxis, np.newaxis]
                # x_wp /= count_[np.newaxis, :, np.newaxis, np.newaxis]
                tt /= count_
                pred = post_proc(pred[np.newaxis].argmax(axis=1)[0])
                # pred = pred[np.newaxis].argmax(axis=1)[0]
                gt_wp = nd.array(gt_wp)
                pred = nd.array(pred[np.newaxis])

                dice_val.append(dice_wp(pred, gt_wp).expand_dims(0))
                loss_val.append(self.loss(pred, gt_wp).expand_dims(0))

                if self.show_val:
                    pred = pred[0].asnumpy()
                    fig = plt.figure(0)
                    for jj in range(pred.shape[0]):
                        plt.subplot(4, np.ceil(pred.shape[0] / 4), jj + 1)
                        plt.imshow(x_wp[-1, jj], cmap='gray', vmin=0, vmax=1)
                        if gt_wp[0, jj].sum() > 0:
                            plt.contour(gt_wp[0, jj].asnumpy(), linewidths=.2)
                        if pred[jj].sum() > 0:
                            plt.contour(pred[jj], colors='r', linewidths=.2)
                        plt.axis('off')
                    plt.savefig(
                        '{:s}/{:04d}_{:02d}_{:.2f}.png'.format(self.dir_fig, epoch, i, dice_val[-1][0].asscalar()),
                        dpi=350)
                    plt.close('all')
                    if self.display_img:
                        plt.show()

            #     if self.file_suffix is "_full":
            #         pred_ = nd.zeros(shape=(pred.shape[0], pred.shape[1], 41, pred.shape[-1], pred.shape[-2]))
            #         pred_[:, :, :pred.shape[2]] = pred
            #         preds.append(pred_)
            #     else:
            #         preds.append(pred)
            # preds = nd.concat(*preds, dim=0).asnumpy()
            # dice_val = self.get_dice_wp(preds)

            logging.info(nd.concat(*dice_val)[0] * 100)
            logging.info((nd.concat(*dice_val)[0] * 100).mean())
        return dice_val, loss_val

    def run(self, lg):
        """Run the training
        lg: logger"""
        global_step = 0
        tic = 0
        best_dice_train, best_dice_val = -1, -1
        best_train_epoch, best_val_epoch = 0, 0
        for epoch in range(self.resumed_epoch+1, self.epochs):
            etic = time.time()
            lg.info("[Epoch {}/{}]".format(epoch, self.epochs))
            self.model.net.hybridize()
            dice_train = []
            loss_train = []
            self.is_val = False
            # TRAIN
            btic = time.time()
            for (i, idx) in enumerate(self.train_iter._batch_sampler):
                if self.val_only:
                    break
                batch = nd.array(self.training_set[idx], ctx=self.ctx)
                gt = batch[:, -1]
                if Training.use_ADC and Training.use_multi_branches:
                    x1 = batch[:, -2]
                    x2 = batch[:, 0, 1, -2]
                    with autograd.record():
                        pred = self.model.net(x1, x2)
                        loss = self.loss(pred, gt)
                else:
                    x = batch[:, :-1]
                    with autograd.record():
                        pred = self.model.net(x)
                        loss = self.loss(pred, gt)
                loss.backward()
                self.Trainer.step(x.shape[0])
                dice_train.append(dice_wp(pred, gt).expand_dims(0))
                loss_train.append(loss.expand_dims(0))
                if self.log_interval and (i + 1) % self.log_interval == 0:
                    current_dice_train = nd.concat(*dice_train)[0].mean().asscalar() * 100
                    lg.info("[Epoch {}/{}] [Batch {}], Speed: {:.3f} samples/sec, "
                            "{}={:.3f}, Dice={:.2f}  [global-step {}] [lr= {:.6f}]".
                            format(epoch, self.epochs, i, self.train_batch_size / (time.time() - btic), self.loss_term,
                                   nd.concat(*loss_train)[0].mean().asscalar(),
                                   current_dice_train, global_step, self.Trainer.learning_rate))
                global_step += 1
                if global_step == self.steps:
                    break
                btic = time.time()
            if global_step == self.steps:
                break

            # VALIDATION
            dice_val, loss_val = self.validate(epoch)

            if self.val_only:
                break
            # Save checkpoints & epoch-level logging
            if epoch >= self.start_validation:
                if current_dice_train > best_dice_train:
                    best_dice_train = current_dice_train
                    best_train_epoch = epoch
                best_dice_val, best_val_epoch = self.model.save_params(epoch, nd.concat(*dice_val)[0].mean().asscalar() * 100, best_dice_val, best_val_epoch, self.Trainer)
                lg.info("[Training]   Loss = {:.3f}, Dice = {:.2f},     [Validation] Loss = {:.3f}, Dice = {:.2f}".format(
                    nd.concat(*loss_train)[0].mean().asscalar(),
                    nd.concat(*dice_train)[0].mean().asscalar() * 100,
                    nd.concat(*loss_val)[0].mean().asscalar(),
                    nd.concat(*dice_val)[0].mean().asscalar() * 100))
                lg.info("Current best dice: [Training] {:.2f} (@ep {:03d}),   [Validataion] {:.2f} (@ep {:03d})".
                        format(best_dice_train, best_train_epoch, best_dice_val, best_val_epoch))

            lg.info("Epoch duration: {:.3f} sec".format(time.time() - etic))
        lg.info("Total training time: {:.3f} s".format(time.time() - tic))


class Model(Training):
    """initialize the network"""
    def build_net(self):
        if Training.use_ADC and Training.use_multi_branches:
            net_init = net_init2
        else:
            net_init = net_init1
        opts = net_init.Init(activation=Training.activation, dense_forward=Training.dense_forward)
        # opts.description()
        self.net = net_init.DenseMultipathNet(opts)

    def review_network(self):
        """inspect the network in details"""
        net_init.review_network(self.net, use_symbol=True, dir_out=self.dir_out, timing=False)

    def init_params(self):
        """initialize network parameters"""
        self.net.collect_params().initialize(initializer.Xavier(magnitude=2), ctx=self.ctx)

    def load_params(self):
        # self.net = gluon.nn.SymbolBlock.imports('{:s}-symbol.json'.format(self.dir_out + self.prefix), ['data'],
        #                                         glob("{:s}_{:04d}_*.params".format(self.dir_out + self.prefix, self.resumed_epoch))[0],
        #                                         ctx=self.ctx)
        self.net.load_parameters(glob("{:s}_{:04d}_*.params".format(self.dir_out + self.prefix, self.resumed_epoch))[0],
                                 ctx=self.ctx)

    def save_params(self, epoch, current_dice, best_dice, best_epoch, trainer):
        """save params at every save_interval interval & at epoch with best dice"""
        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = epoch
            self.net.save_parameters("{:s}_best.params".format(self.dir_out + self.prefix))
            # self.net.export("{:s}_best".format(self.dir_out + self.prefix))
            # trainer.save_states("{:s}_best.states".format(self.dir_out + self.prefix))
        if self.save_interval and ((epoch+1) % self.save_interval == 0):
            self.net.save_parameters("{:s}_{:04d}_{:.2f}.params".format(self.dir_out + self.prefix, epoch, current_dice))
        if epoch == 0:
            self.net.export("{:s}".format(self.dir_out + self.prefix))
            # trainer.save_states("{:s}_{:04d}.states".format(self.dir_out + self.prefix, epoch))
        return best_dice, best_epoch


if __name__ == "__main__":
    trn = Training(get_args())
    trn.fix_seed()
    trn.prepare_dirs()
    lg = trn.set_logger()
    trn.list_parameters(lg)
    lg.info('Loading data... ')
    trn.load_data()
    # lg.info('Transforming data dimensions...')
    # trn.transform_data()
    lg.info('Splitting dataset to training and validation sets [%d/%d]...' % (trn.train_amount, trn.val_amount))
    trn.split_data()
    # lg.info('Excluding samples...')
    # trn.exclude_samples()
    lg.info('Setup data iterators...')
    trn.set_iter()
    lg.info('Setup training context...')
    trn.set_context()
    lg.info('Constructing the network (Dense Multi-path Net)...')
    trn.model = Model(get_args())
    trn.model.build_net()
    if trn.to_review_network:
        trn.model.review_network()
    if trn.resumed_epoch < 0:
        lg.info('Initialize network parameters')
        trn.model.init_params()
    else:
        trn.model.load_params()
    lg.info('Setup trainer...')
    trn.set_trainer()

    lg.info('Setup loss metric...')
    trn.set_loss()
    lg.info('Training data size: %s' % (trn.im[:, :-1].shape,))
    lg.info('Training ground truth size: %s' % (trn.im[:, -1].shape,))

    if trn.coor_maps:
        lg.info('Get coordinate channels...')
        trn.get_coor_maps()

    lg.info('Start training now...')
    trn.run(lg)
    lg.info('Training done.')