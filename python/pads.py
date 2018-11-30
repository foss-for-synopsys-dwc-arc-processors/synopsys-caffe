import caffe
import numpy as np


class Pad(caffe.Layer):
    """
    Pad a tensor using the numpy.pad() method.
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        self.paddings = eval(self.param_str)

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = bottom[0].data.shape
        shape += np.sum(self.paddings, axis=1)
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.pad(
            bottom[0].data, pad_width=self.paddings, mode="constant")

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]


class PadV2(caffe.Layer):
    """
    Pad a tensor using the numpy.pad() method.
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        params = eval(self.param_str)
        self.paddings = params["paddings"]
        self.constant_values = params["constant_values"]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = bottom[0].data.shape
        shape += np.sum(self.paddings, axis=1)
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.pad(
            bottom[0].data,
            pad_width=self.paddings,
            mode="constant",
            constant_values=self.constant_values
        )

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]


class MirrorPad(caffe.Layer):
    """
    Pad a tensor using the numpy.pad() method.
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        param_str = eval(self.param_str)
        self.paddings = param_str["paddings"]
        self.mode = param_str["mode"].lower()

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = bottom[0].data.shape
        shape += np.sum(self.paddings, axis=1)
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.pad(
            bottom[0].data, pad_width=self.paddings, mode=self.mode)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
