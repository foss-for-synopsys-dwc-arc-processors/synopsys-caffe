import caffe
import numpy as np


class StackTensors(caffe.Layer):
    """
    Implementation of tf.stack
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) < 1:
            raise Exception("More than one Tensor is needed!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        shape = set()
        for i in range(len(bottom)):
            shape.add(bottom[i].data.shape)
        if len(shape) > 1:
            raise Exception("Inputs must have the same shape")
        self.axis = int(self.param_str)
        self.out_shape = list(shape.pop())
        self.out_shape.insert(self.axis, len(bottom))

    def reshape(self, bottom, top):
        top[0].reshape(*self.out_shape)

    def forward(self, bottom, top):
        if bottom[0].data.ndim == 0:
            inputs = [bottom[i].data.tolist() for i in range(len(bottom))]
            top[0].data[...] = inputs
        else:
            inputs = [bottom[i].data for i in range(len(bottom))]
            top[0].data[...] = np.stack(inputs, axis=self.axis)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
