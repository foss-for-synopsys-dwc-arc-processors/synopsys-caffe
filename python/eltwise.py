import caffe
import numpy as np


class Eltwise(caffe.Layer):
    """
    Get a tensor's slice: implementation of element-wise operatios with broadcasting
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
        self.coeff = d["coeff"]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0 or bottom[1].count == 0:
            raise Exception("Input must not be empty!")
        if bottom[0].data.ndim > bottom[1].data.ndim:
            top[0].reshape(*bottom[0].shape)
        else:
            top[0].reshape(*bottom[1].shape)

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

    def backward(self, top, propagate_down, bottom):
        pass
