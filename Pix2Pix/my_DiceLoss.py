import mxnet as mx
import numpy as np


class DiceLoss(mx.operator.CustomOp):
    """
    Compute energy based on dice coefficient.
    """
    union = None
    intersection = None
    result = None
    gt = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the dice. the result of the softmax and the ground truth.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2 * bottom[1].count:
            # print bottom[0].shape
            # print bottom[1].shape
            raise Exception("the dimension of inputs should match")

        # loss output is two scalars (mean and std)
        top[0].reshape(1)

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])))
        y /= y.sum(axis=1).reshape((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]))

        bottom = [None] * 2
        bottom[0] = y  # softmax
        bottom[1] = in_data[1].asnumpy()  # label

        dice = np.zeros(bottom[0].shape[0], dtype=np.float32)
        self.union = np.zeros(bottom[0].shape[0], dtype=np.float32)
        self.intersection = np.zeros(bottom[0].shape[0], dtype=np.float32)

        self.result = np.reshape(np.squeeze(np.argmax(bottom[0][...], axis=1)),
                                 [bottom[0].shape[0], np.prod(bottom[0].shape[2:])])
        self.gt = np.reshape(np.squeeze(bottom[1][...]), [bottom[1].shape[0], np.prod(bottom[1].shape[1:])])

        # self.gt = (self.gt > 0.5).astype(dtype=np.float32)
        # self.result = self.result.astype(dtype=np.float32)

        for i in range(0, bottom[0].shape[0]):
            # compute dice
            CurrResult = (self.result[i, :]).astype(dtype=np.float32)
            CurrGT = (self.gt[i, :]).astype(dtype=np.float32)

            self.union[i] = (np.sum(CurrResult) + np.sum(CurrGT))
            self.intersection[i] = (np.sum(CurrResult * CurrGT))

            dice[i] = 2 * self.intersection[i] / (self.union[i] + 0.00001)
            # print dice[i]

        # top[0][0] = np.sum(dice)
        # out_data[0] = np.sum(dice)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        bottom = [None] * 2
        bottom[0] = out_data[0].asnumpy()  # prob (output of forward)
        bottom[1] = in_data[1].asnumpy()  # label

        for btm in [0]:
            prob = bottom[btm][...].reshape([bottom[0].shape[0], bottom[0].shape[1],
                                             np.prod(bottom[0].shape[2:])])
            diff = np.zeros(prob.shape, dtype=np.float32)
            for i in range(0, prob.shape[0]):
                # diff[i, 0, :] += 2.0 * (
                #     (self.gt[i, :] * self.union[i]) / ((self.union[i]) ** 2) - 2.0 * prob[i, 1, :] * (
                #     self.intersection[i]) / (
                #         (self.union[i]) ** 2) + 1e-12)
                # diff[i, 1, :] -= 2.0 * (
                #     (self.gt[i, :] * self.union[i]) / ((self.union[i]) ** 2) - 2.0 * prob[i, 1, :] * (
                #     self.intersection[i]) / (
                #         (self.union[i]) ** 2) + 1e-12)
                diff[i, 0, :] += 2.0 * ((self.gt[i, :] * self.union[i]) - 2.0 * prob[i, 1, :] *
                                        (self.intersection[i])) / ((self.union[i]) ** 2 + 1e-12)
                diff[i, 1, :] -= 2.0 * ((self.gt[i, :] * self.union[i]) - 2.0 * prob[i, 1, :] *
                                        (self.intersection[i])) / ((self.union[i]) ** 2 + 1e-12)
        diff = diff.reshape([bottom[0].shape[k] for k in np.arange(bottom[0].ndim)])
        self.assign(in_grad[0], req[0], mx.nd.array(diff))
        # pass

@mx.operator.register("dice_loss")
class DiceLossProtoc(mx.operator.CustomOpProp):
    def __init__(self):
        super(DiceLossProtoc, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label1']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return DiceLoss()

