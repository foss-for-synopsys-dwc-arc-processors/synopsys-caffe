import caffe
import numpy as np


class Eltwise(caffe.Layer):
    """Get a tensor's slice: implementation of element-wise operations
    with broadcasting
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        # bottom[1] is begin
        # bottom[2] is size
        if len(bottom) != 2:
            raise Exception("Please input two Tensors!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        d = eval(self.param_str)
        self.operation = d["operation"]
        # only SUM supports coeff
        if self.operation == 1:
            self.coeff = d["coeff"]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0 or bottom[1].count == 0:
            raise Exception("Input must not be empty!")
        b0 = bottom[0].data
        b1 = bottom[1].data
        min_dim = min(b0.ndim, b1.ndim)
        if b0.ndim >= b1.ndim:
            shape = list(b0.shape)
            min_dim = b1.ndim
        else:
            shape = list(b1.shape)
            min_dim = b0.ndim
        for i in range(-min_dim, 0, 1):
            shape[i] = max(bottom[0].data.shape[i], bottom[1].data.shape[i])

        top[0].reshape(*shape)

    def forward(self, bottom, top):
        if self.operation == 0:
            top[0].data[...] = (np.array(bottom[0].data) *
                                np.array(bottom[1].data))
        elif self.operation == 1:
            top[0].data[...] = (self.coeff[0]*np.array(bottom[0].data) +
                                self.coeff[1]*np.array(bottom[1].data))
        elif self.operation == 2:
            top[0].data[...] = np.maximum(np.array(bottom[0].data),
                                          np.array(bottom[1].data))
        elif self.operation == 3:
            top[0].data[...] = np.divide(np.array(bottom[0].data),
                                         np.array(bottom[1].data))
        elif self.operation == 4:
            top[0].data[...] = np.minimum(np.array(bottom[0].data),
                                          np.array(bottom[1].data))

    def backward(self, top, propagate_down, bottom):
        pass
