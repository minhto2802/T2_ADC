"""Training the network for prostate segmentation"""
from mxnet import gluon, io, gpu, initializer, autograd
from gluoncv import utils as gutils
from utils.init_params import InitParams
from networks import dmnet_gluon_init as net_init
from utils.utils import *
import numpy as np
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
    parser.add_argument('--seed', type=int,
                        help='A number specifying which fixed seed is used for MXNet, Numpy and Python')
    parser.add_argument('--log_interval', type=int, help='Logging interval, default: 5')
    parser.add_argument('--save_interval', type=int, help='Number of epochs between each checkpoints, default: 1')
    parser.add_argument('--val_interval', type=int, help='Number of epochs between each validation, default: 1')
    parser.add_argument('--prefix', type=str, help='Checkpoint name prefix, default: "dmnet"')
    args = parser.parse_args()
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

    def load_data(self):
        """load inputs"""
        # a = np.load(r'F:\Minh\projects\T2_ADC\CycleGAN\results\T2_ADC_run11\test_49_TrainingSet/ps_input.npy')
        # b = np.load(r'F:\Minh\projects\T2_ADC\CycleGAN\results\T2_ADC_run11\test_49_TrainingSet/ps_input_idx.npy')
        # self.im = np.load("%s%s.npy" % (self.dir_in, self.training_data_name))
        # self.lab = np.load("%s%s.npy" % (self.dir_in, self.training_gt_name))
        self.im = np.load(r'%s/T2_ADC.npy' % self.dir_in)
        self.lab = np.squeeze(np.load(r'%s/T2_ADC_lab.npy' % self.dir_in))
        self.idx = np.load(r'%s/T2_ADC_idx.npy' % self.dir_in)

    def split_data(self):
        """split dataset into training & validation"""
        # splits = np.loadtxt('%s/splits.txt' % self.dir_in, 'int')
        # self.train_idx = splits[self.split][:self.train_amount]
        # self.val_idx = splits[self.split][self.train_amount:(self.train_amount + self.val_amount)]
        splits = np.load('%s/T2_ADC_val_idx.npy' % self.dir_in)[self.split]
        self.train_idx = np.where(splits == 0)[0]
        self.val_idx = np.where(splits == 1)[0]

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

    def set_iter(self):
        """Prepare data iterator for training and validation"""
        self.train_iter = io.NDArrayIter(self.im[self.train_idx],
                                         self.lab[self.train_idx],
                                         batch_size=self.train_batch_size, shuffle=True)
        self.val_iter = io.NDArrayIter(self.im[self.val_idx],
                                       self.lab[self.val_idx],
                                       batch_size=self.val_batch_size)

    def set_context(self):
        """set up context (GPUs or CPUs)"""
        # self.ctx = [gpu(int(i)) for i in self.gpus.split(',') if i.strip()][0]
        # self.ctx = self.ctx if self.ctx else [cpu()]
        self.ctx = gpu(self.gpu_id)

    def set_trainer(self):
        """Set up training policy"""
        self.optimizer_params = {'learning_rate': self.base_lr,
                                 'wd': self.wd,
                                 'clip_gradient': None,
                                 # 'rescale_grad': 1.0 / len(self.gpu_id) if len(self.gpu_id) > 0 else 1.0}
                                 }
        self.Trainer = gluon.Trainer(self.model.net.collect_params(), optimizer=self.optimizer,
                                     optimizer_params=self.optimizer_params)

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

    def run(self, lg):
        """Run the training
        lg: logger"""
        global_step = 0
        tic = 0
        best_dice = -1
        for epoch in range(self.epochs):
            etic = time.time()
            lg.info("[Epoch {}/{}]".format(epoch, self.epochs))
            self.model.net.hybridize()
            self.train_iter.reset()
            self.val_iter.reset()
            dice_train = []
            dice_val = []
            loss_train = []
            loss_val = []
            # TRAIN
            btic = time.time()
            for (i, batch) in enumerate(self.train_iter):
                x = batch.data[0].as_in_context(self.ctx)
                gt = batch.label[0].as_in_context(self.ctx)
                with autograd.record():
                    pred = self.model.net(x)
                    loss = self.loss(pred, gt)
                    loss.backward()
                self.Trainer.step(self.train_batch_size)
                dice_train.append(dice_wp(pred, gt).expand_dims(0))
                loss_train.append(loss.expand_dims(0))
                if self.log_interval and (i + 1) % self.log_interval == 0:
                    lg.info("[Epoch {}/{}] [Batch {}], Speed: {:.3f} samples/sec, "
                            "{}={:.3f}, Dice={:.2f}  [global-step {}]".
                            format(epoch, self.epochs, i, self.train_batch_size / (time.time() - btic), self.loss_term,
                                   nd.concat(*loss_train)[0].mean().asscalar(),
                                   nd.concat(*dice_train)[0].mean().asscalar() * 100, global_step))
                global_step += 1
                if global_step == self.steps:
                    break
                btic = time.time()

            # VALIDATION
            if self.val_interval and (epoch + 1) % self.val_interval == 0:
                for (i, batch) in enumerate(self.val_iter):
                    x = batch.data[0].as_in_context(self.ctx)
                    gt = batch.label[0].as_in_context(self.ctx)
                    pred = self.model.net(x)
                    dc = dice_wp(pred, gt)
                    dice_val.append(dice_wp(pred, gt).expand_dims(0))
                    loss_val.append(self.loss(pred, gt).expand_dims(0))

            # Save checkpoints & epoch-level logging
            best_dice = self.model.save_params(epoch, nd.concat(*dice_val)[0].mean().asscalar() * 100, best_dice)
            lg.info("[Training]   Loss = {:.3f}, Dice = {:.2f},     [Validation] Loss = {:.3f}, Dice = {:.2f}".format(
                nd.concat(*loss_train)[0].mean().asscalar(),
                nd.concat(*dice_train)[0].mean().asscalar() * 100,
                nd.concat(*loss_val)[0].mean().asscalar(),
                nd.concat(*dice_val)[0].mean().asscalar() * 100))
            lg.info("Current best dice: {:.2f}".format(best_dice))
            lg.info("Epoch duration: {:.3f} sec".format(time.time() - etic))
        lg.info("Total training time: {:.3f} s".format(time.time() - tic))


class Model(Training):
    """initialize the network"""
    opts = net_init.Init()
    # opts.description()
    net = net_init.DenseMultipathNet(opts)

    def review_network(self):
        """inspect the network in details"""
        net_init.review_network(self.net, use_symbol=True, dir_out=self.dir_out, timing=False)

    def init_params(self):
        """initialize network parameters"""
        self.net.collect_params().initialize(initializer.Xavier(magnitude=2.2), ctx=self.ctx)

    def save_params(self, epoch, current_dice, best_dice):
        """save params at every save_interval interval & at epoch with best dice"""
        if current_dice > best_dice:
            best_dice = current_dice
            self.net.save_parameters("{:s}_best.params".format(self.dir_out + self.prefix))
        if self.save_interval and ((epoch+1) % self.save_interval == 0):
            self.net.save_parameters("{:s}_{:04d}_{:.2f}.params".format(self.dir_out + self.prefix, epoch, current_dice))
        return best_dice


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
    lg.info('Setup data iterators...')
    trn.set_iter()
    lg.info('Setup training context...')
    trn.set_context()
    lg.info('Constructing the network (Dense Multi-path Net)...')
    trn.model = Model(get_args())
    if trn.to_review_network:
        trn.model.review_network()
    lg.info('Initialize network parameters')
    trn.model.init_params()
    lg.info('Setup trainer...')
    trn.set_trainer()
    lg.info('Setup loss metric...')
    trn.set_loss()
    lg.info('Training data size: %s' % (trn.im.shape,))
    lg.info('Training ground truth size: %s' % (trn.lab.shape,))

    lg.info('Start training now...')
    trn.run(lg)
    lg.info('Training done.')
