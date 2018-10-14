"""gluon-style implementation of dmnet"""
from mxnet import nd, initializer, gpu, symbol, viz
from mxnet.gluon.nn import Conv2D, Conv2DTranspose, Activation, BatchNorm, HybridSequential, \
    HybridBlock, Dropout, MaxPool2D, LeakyReLU
from math import floor
import inspect
import time
import numpy as np


class Init:
    """initialize parameters"""
    def __init__(self, units=[6, 12, 24, 48], num_stage=4, reduction=.5, init_channels=64, growth_rate=4,
                 bottle_neck=True, drop_out=0, bn_mom=.1, bn_eps=1e-5, work_space=1024, num_classes=1,
                 activation='relu', use_bias=False, num_fpg=-1, use_softmax=True, quantization_levels=None,
                 last_act=True):
        self.units = units
        self.num_stage = num_stage
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.num_fpg = num_fpg  # number of feature maps per group
        self.init_channels = init_channels
        self.bottle_neck = bottle_neck
        self.drop_out = drop_out
        self.bn_mom = bn_mom
        self.bn_eps = bn_eps
        self.work_Space = work_space
        self.activation = activation
        self.use_bias = use_bias
        self.use_softmax = use_softmax
        self.quantization_levels = quantization_levels
        self.num_classes = num_classes
        self.last_act = last_act

    def description(self):
        """List all parameters"""
        L = inspect.getmembers(self)
        for l in L:
            if '__' not in l[0] and l[0] != 'description':
                print('%15s: %s' % (l[0], l[1]))


class FirstBlock(HybridBlock):
    """Return FirstBlock for building DenseNet"""
    def __init__(self, opts):
        super(FirstBlock, self).__init__()
        self.fblock = HybridSequential()
        self.fblock.add(Conv2D(channels=opts.init_channels, kernel_size=(3, 3),
                               strides=(1, 1), padding=(1, 1), use_bias=opts.use_bias))

        # self.fblock.add(BatchNorm(momentum=opts.bn_mom, epsilon=opts.bn_eps))
        # self.fblock.add(Activation(opts.activation))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.fblock(x)


class BasicBlock(HybridBlock):
    """Return BaiscBlock Unit for building DenseBlock
    Parameters
    ----------
    opts: instance of Init
    """
    def __init__(self, opts):
        super(BasicBlock, self).__init__()
        self.bblock = HybridSequential()
        if opts.bottle_neck:
            self.bblock.add(BatchNorm(momentum=opts.bn_mom, epsilon=opts.bn_eps))
            if not opts.trans_block:
                self.bblock.add(LeakyReLU(alpha=.2))
            else:
                self.bblock.add(Activation(opts.activation))
            self.bblock.add(Conv2D(channels=int(opts.growth_rate * 4), kernel_size=(1, 1),
                                   strides=(1, 1), use_bias=opts.use_bias, padding=(0, 0)))
            if opts.drop_out > 0:
                self.bblock.add(Dropout(opts.drop_out))
        self.bblock.add(BatchNorm(momentum=opts.bn_mom, epsilon=opts.bn_eps))
        self.bblock.add(Activation(activation=opts.activation))
        self.bblock.add(Conv2D(channels=int(opts.growth_rate), kernel_size=(3, 3),
                               strides=(1, 1), use_bias=opts.use_bias, padding=(1, 1)))
        if opts.drop_out > 0:
            self.bblock.add(Dropout(opts.drop_out))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        # print(self.bblock(x).shape)
        # print(x.shape)
        return F.Concat(x, self.bblock(x))


class DenseBlock(HybridBlock):
    """Return DenseBlock Unit for building DenseNet
    Parameters
    ----------
    opts: instance of Init
    units_num : int
        the number of BasicBlock in each DenseBlock
    """
    def __init__(self, opts, units_num):
        super(DenseBlock, self).__init__()
        self.dblock = HybridSequential()
        for i in range(units_num):
            self.dblock.add(BasicBlock(opts))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        return self.dblock(x)


class TransitionBlock(HybridBlock):
    """Return TransitionBlock Unit for building DenseNet
    Parameters
    ----------
    num_stage : int
        Number of stage
    num_filters : int
        Number of output channels
    """
    def __init__(self, opts, num_filters, pool_type='avg'):
        super(TransitionBlock, self).__init__()
        self.pool_type = pool_type
        self.tblock = HybridSequential()
        self.tblock.add(BatchNorm(momentum=opts.bn_mom, epsilon=opts.bn_eps))
        self.tblock.add(Activation(opts.activation))
        self.tblock.add(Conv2D(channels=int(num_filters * opts.reduction), kernel_size=(1, 1),
                               strides=(1, 1), use_bias=opts.use_bias, padding=(0, 0)))
        if opts.drop_out > 0:
            self.tblock.add(Dropout(opts.drop_out))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        return F.Pooling(self.tblock(x), global_pool=False, kernel=(2, 2), stride=(2, 2), pool_type=self.pool_type)


def conv_factory(opts, num_filters, kernel_size, group=1):
    """A convenience function for convolution with batchnorm & activation"""
    pad = int((kernel_size - 1) / 2)
    out = HybridSequential()
    out.add(BatchNorm(momentum=opts.bn_mom, epsilon=opts.bn_eps))
    out.add(Activation(opts.activation))

    out.add(Conv2D(channels=num_filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1), use_bias=opts.use_bias,
                   padding=(pad, pad), groups=group))
    return out


class ResDBlock(HybridBlock):
    """Residual decoding block"""
    def __init__(self, opts, num_filters, group=1):
        super(ResDBlock, self).__init__()
        if opts.num_fpg != -1:
            group = int(num_filters / opts.num_fpg)
        self.body = HybridSequential()
        with self.body.name_scope():
            self.body.add(conv_factory(opts, num_filters, kernel_size=1))
            self.body.add(conv_factory(opts, num_filters, kernel_size=3, group=group))
            self.body.add(conv_factory(opts, num_filters, kernel_size=1))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        return F.concat(self.body(x), x)


class EncoderBlock(HybridBlock):
    """Return a block in Encoder"""
    def __init__(self, opts, num_unit, num_filters, trans_block=True):
        super(EncoderBlock, self).__init__()
        self.eblock = HybridSequential()
        if trans_block:
            self.eblock.add(TransitionBlock(opts, num_filters=num_filters))
        else:
            self.eblock.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding=(1, 1)))
        opts.trans_block = trans_block
        self.eblock.add(DenseBlock(opts, num_unit))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        return self.eblock(x)


class DecoderBlock(HybridBlock):
    """Return a block in Decoder"""
    def __init__(self, opts, num_filters, res_block=True, factor=1, group=1):
        super(DecoderBlock, self).__init__()
        self.dcblock = HybridSequential()
        if res_block:
            self.dcblock.add(ResDBlock(opts, num_filters * 4, group=group))
        self.dcblock.add(BatchNorm(momentum=opts.bn_mom, epsilon=opts.bn_eps))
        self.dcblock.add(Activation(opts.activation))
        self.dcblock.add(Conv2DTranspose(channels=int(num_filters / factor), kernel_size=(2, 2),
                                         strides=(2, 2), padding=(0, 0), use_bias=opts.use_bias))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        out = self.dcblock(x)
        return out


class EncoderDecoderUnit(HybridBlock):
    """Return a recursive pair of encoder - decoder"""
    def __init__(self, opts, num_filters, stage, inner_block=None, innermost=False):
        super(EncoderDecoderUnit, self).__init__()

        factor = 2 if stage == 0 else 1
        encoder = EncoderBlock(opts, opts.units[stage], num_filters, trans_block=False if stage == 0 else True)
        decoder = DecoderBlock(opts, num_filters, res_block=(not innermost), factor=factor)
        if innermost:
            model = [encoder, decoder]
        else:
            model = [encoder, inner_block, decoder]

        self.net = HybridSequential()
        for block in model:
            self.net.add(block)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        out = F.concat(x, self.net(x))
        return out


class Softmax(HybridBlock):
    """"Softmax"""
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        """Forward"""
        return F.softmax(x, axis=1)


class DenseMultipathNet(HybridBlock):
    """Return a whole network"""
    def __init__(self, opts):
        super(DenseMultipathNet, self).__init__()
        opts.units = opts.units[:opts.num_stage]
        assert (len(opts.units) == opts.num_stage)

        num_filters = opts.init_channels
        num_filters_list = []
        for stage in range(opts.num_stage):
            num_filters += opts.units[stage] * opts.growth_rate
            num_filters = int(floor(num_filters * opts.reduction))
            num_filters_list.append(num_filters)

        self.net = HybridSequential()
        with self.net.name_scope():
            self.blocks = EncoderDecoderUnit(opts, num_filters_list[opts.num_stage-1], opts.num_stage-1, innermost=True)
            for stage in range(opts.num_stage-2, -1, -1):
                self.blocks = EncoderDecoderUnit(opts, num_filters_list[stage], stage, inner_block=self.blocks)
            self.net.add(FirstBlock(opts))
            self.net.add(self.blocks)
            self.net.add(ResDBlock(opts, num_filters=16))
            self.net.add(Conv2D(kernel_size=(1, 1), channels=opts.num_classes, use_bias=opts.use_bias))
            if opts.num_classes > 1:
                self.net.add(BatchNorm(momentum=opts.bn_mom, epsilon=opts.bn_eps))
                if opts.use_softmax:
                    self.net.add(Softmax())
            if opts.last_act:
                self.net.add(Activation('tanh'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        return self.net(x)


def review_network(net, use_symbol=False, timing=True, num_rep=100, dir_out='', print_model_size=False):
    """inspect the network architecture & input - output
    use_symbol: set True to inspect the network in details
    timing: set True to estimate inference time of the network
    num_rep: number of inference"""
    shape = (1, 2, 160, 160)
    if use_symbol:
        x = symbol.Variable('data')
        y = net(x)
        if print_model_size:
            get_model_size(y, to_print=False, data_shape=shape)
        viz.plot_network(y, shape={'data': shape}, node_attrs={"fixedsize": "false"}).view('%sDenseMultipathNet' % dir_out)
    else:
        x = nd.random_normal(0.1, 0.02, shape=shape, ctx=ctx)
        net.collect_params().initialize(initializer.Xavier(magnitude=2), ctx=ctx)

        if timing:
            s1 = time.time()
            y = net(x)
            y.wait_to_read()
            print("First run: %.5f" % (time.time()-s1))

            net.hybridize()
            import numpy as np
            times = np.zeros(num_rep)
            for t in range(num_rep):
                x = nd.random_normal(0.1, 0.02, shape=shape, ctx=ctx)
                s2 = time.time()
                y = net(x)
                y.wait_to_read()
                times[t] = time.time() - s2
            print("Run with hybrid network: %.5f" % times.mean())
        else:
            y = net(x)
        print("Input size: ", x.shape)
        print("Output size: ", y.shape)


def get_model_size(symbol, data_shape=(1, 2, 24, 160, 160), to_print=False):
    """Estimate number of parameters in the networks_"""
    la = symbol.list_arguments()
    # s = symbol.infer_shape(data=data_shape, label1=label_shape)[0]
    s = symbol.infer_shape_partial(data=data_shape)[0]
    total = 0
    for i in np.arange(s.__len__()):
        if to_print:
            print('%s:   %s' % (la[i], s[i]))
        if la[i] == 'data' or la[i] == 'label1' or la[i] == 'lab1':
            continue
        total = total + np.prod(np.array(list(s[i])))
    print('Network size is:  %.6f Mil params' % (total / 1e6))
    return total


if __name__ == "__main__":
    ctx = gpu()
    opts = Init(num_fpg=-1)
    opts.description()

    net = DenseMultipathNet(opts)
    review_network(net, use_symbol=True, timing=False, print_model_size=True)


