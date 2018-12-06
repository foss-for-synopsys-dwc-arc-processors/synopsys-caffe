import caffe
import numpy as np


class Mean(caffe.Layer):
    """
    implement tf.reduce_mean()
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        params = eval(self.param_str)
        self.axis = params["axis"]
        self.keepdims = params["keepdims"]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = list(bottom[0].data.shape)
        for i in reversed(sorted(self.axis)):
            if self.keepdims:
                shape[i] = 1
            else:
                del shape[i]
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.mean(
            bottom[0].data, axis=self.axis, keepdims=self.keepdims)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]


class Prod(caffe.Layer):
    """
    implement tf.reduce_prod()
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        params = eval(self.param_str)
        self.axis = params["axis"]
        self.keepdims = params["keepdims"]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = list(bottom[0].data.shape)
        for i in reversed(sorted(self.axis)):
            if self.keepdims:
                shape[i] = 1
            else:
                del shape[i]
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.prod(
            bottom[0].data, axis=self.axis, keepdims=self.keepdims)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]


class Sum(caffe.Layer):
    """
    implement tf.reduce_sum()
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        params = eval(self.param_str)
        self.axis = params["axis"]
        self.keepdims = params["keepdims"]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = list(bottom[0].data.shape)
        for i in reversed(sorted(self.axis)):
            if self.keepdims:
                shape[i] = 1
            else:
                del shape[i]
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.sum(
            bottom[0].data, axis=self.axis, keepdims=self.keepdims)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
