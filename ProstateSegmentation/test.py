"""Test the network for prostate segmentation"""
from networks import dmnet_gluon_init as net_init
from mxnet import gpu, io
from utils.utils import *
import logging
import argparse
import inspect
import time
from utils.init_params import InitParams


def get_args():
    """get commandline parameters"""
    parser = argparse.ArgumentParser(description='Run CNN prediction on prostate segmentation.')
    parser.add_argument('--run_id', type=int, help='Training sessions ID', default=999999)
    parser.add_argument('--test_batch_size', type=int, help='Size of each mini-batch in test', default=1)
    parser.add_argument('--dir_in', type=str, help='Path to input file',
                        default=r"F:\Minh\mxnet\projects\prostate_segmentation\inputs\NIH/")
    parser.add_argument('--dir_out', type=str, help='Path to save training session')
    parser.add_argument('--prefix', type=str, help='Checkpoint name prefix, default: "dmnet"', default='dmnet')
    parser.add_argument('--dataset', type=str, help='Path to save training session', default='NIH')
    parser.add_argument('--split', type=int, help='Split index', default=1)
    parser.add_argument('--gpus', type=str, help='GPUs used for training, e.g. "0, 2", '
                                                 'leave empty to use CPUs, default: "0"', default="1")
    parser.add_argument('--log_file', type=str, help='Name of the log file, default "log.txt"', default='log_test.txt')
    parser.add_argument('--training_data_name', type=str, default="im_normed",
                        help='Name of the training image file (include both training and validation set')
    parser.add_argument('--training_gt_name', type=str, default="lab",
                        help='Name of the training ground truth file (include both training and validation set')
    args = parser.parse_args()
    args.dir_out = "F:\BACKUPS\%s\outputs/run%d/" % (args.dataset, args.run_id)
    return args


class Test(InitParams):
    """Perform the prostate segmentation on the test set"""
    def __init__(self):
        self.set_context()
        self.get_args()

    def get_args(self):
        """Assign arguments from command lines to the Training class"""
        args = get_args()
        for key, value in zip(args.__dict__.keys(), args.__dict__.values()):
            if value is not None:
                self.__setattr__(key, value)

    def list_parameters(self, logger):
        """List and log all current parameters"""
        params = inspect.getmembers(self)
        logger.info("List all parameters...")
        for param in params:
            if not param[0].__contains__('__') and not callable(param[1]):
                if param[1] is not None:
                    logger.info("   {}: {}".format(param[0], param[1]))

    def load_data(self):
        """load inputs"""
        self.im = np.load("%s%s.npy" % (self.dir_in, self.training_data_name))
        self.lab = np.load("%s%s.npy" % (self.dir_in, self.training_gt_name))

    def split_data(self):
        """split dataset into training & validation"""
        splits = np.loadtxt('%s/splits.txt' % self.dir_in, 'int')
        self.test_idx = splits[self.split][-50:]

    def transform_data(self):
        """transpose images to NCDHW (im) and NDHW (lab)"""
        self.split_data()
        self.im = np.transpose(self.im, (4, 3, 2, 0, 1))[self.test_idx]
        self.lab = np.transpose(self.lab, (3, 2, 0, 1))[self.test_idx]

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
        """Prepare data iterator for test"""
        self.test_iter = io.NDArrayIter(self.im, self.lab, batch_size=self.test_batch_size)

    def set_context(self):
        """set up context (GPUs or CPUs)"""
        # self.ctx = [gpu(int(i)) for i in self.gpus.split(',') if i.strip()][0]
        # self.ctx = self.ctx if self.ctx else [cpu()]
        self.ctx = gpu(1)

    def run(self):
        """Run the prediction on test set"""
        preds = []
        self.model.net.hybridize()
        for (i, batch) in enumerate(self.test_iter):
            pred = self.model.net(batch.data[0].as_in_context(self.ctx)).argmax(axis=1)
            preds.append(pred)

        preds = nd.concat(*preds, dim=0)[:self.im.shape[0]]
        dice = dice_wp(preds, nd.array(self.lab, ctx=self.ctx)).asnumpy()
        lg.info('Dice per subject: ')
        lg.info('\n{}'.format(dice))
        lg.info('Mean Dice: %.2f', dice.mean() * 100)
        lg.info('Total test time: %.4f' % (time.time() - tic))
        lg.info('Save output data to %s...' % self.dir_out)
        np.save(self.dir_out + 'test_pred.npy', preds.asnumpy())
        lg.info('Done.')


class Model(Test):
    """initialize the network"""
    opts = net_init.Init()
    # opts.description()
    net = net_init.DenseMultipathNet(opts)

    def review_network(self):
        """inspect the network in details"""
        net_init.review_network(self.net, use_symbol=True, dir_out=self.dir_out, timing=False)

    def load_params(self):
        """initialize network parameters"""
        self.net.load_params(filename=self.dir_out + self.prefix + '_best.params', ctx=self.ctx)


if __name__ == "__main__":
    tic = time.time()
    test = Test()
    lg = test.set_logger()
    test.list_parameters(lg)
    lg.info('Loading data...')
    test.load_data()
    lg.info('Transforming data dimensions...')
    test.transform_data()
    lg.info('Setup context...')
    test.set_context()
    lg.info('Setup data iterators...')
    test.set_iter()

    test.model = Model()
    lg.info('Initialize network parameters')
    test.model.load_params()

    test.run()
