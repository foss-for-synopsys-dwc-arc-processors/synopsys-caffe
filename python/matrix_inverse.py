import caffe
import numpy as np
from numpy.linalg import inv

class MatrixInverse(caffe.Layer):
    """
    implement tf.matrix_inverse/tf.linalg.inv
    ref: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.inv.html
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
        shape = list(bottom[0].data.shape)
        if shape[2] != shape[3]:
            raise Exception("Last 2 dimensions of the array must be square!")
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = inv(bottom[0].data)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
