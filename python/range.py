import caffe
import numpy as np


class Range(caffe.Layer):
    """
    Implemention of tf.range() method.
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        #if len(bottom) != 1:
        #    raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        d = eval(self.param_str)
        if d["start"] != None:
            self.start = d["start"]
        else:
            self.start = 0
        self.limit = d["limit"]
        if d["delta"] != None:
            self.delta = d["delta"]
        else:
            self.delta = 1

    def reshape(self, bottom, top):
        # check input dimensions
        #if bottom[0].count == 0:
        #    raise Exception("Input must not be empty!")
        length = (self.limit - self.start) / self.delta
        top[0].reshape(length)

    def forward(self, bottom, top):
        top[0].data[...] = np.arange(self.start, self.limit, self.delta)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
