import caffe
import numpy as np


class Reshape(caffe.Layer):
    """
    implementation of the tf.reshape() with two inputs.
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        # bottom[1] is blob to be reshaped
        # bottom[2] is shape
        if len(bottom) != 2:
            raise Exception("Please input two Tensors!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = [int(i) for i in bottom[1].data]
        if -1 in shape:
            total = np.prod(bottom[0].data.shape)
            for i in shape:
                if -1 != i:
                    total = total/i
                else:
                    d = i
            shape[d] = total 
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data.ravel()

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
