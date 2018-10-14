import caffe
import numpy as np


class TensorShape(caffe.Layer):
    """
    Get a tensor's shape using the numpy.shape() method.
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        top[0].reshape(len(bottom[0].data.shape))

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data.shape

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
