import caffe
import numpy as np


class PadTensor(caffe.Layer):
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
        self.pad_t = int(params["pad_t"])
        self.pad_b = int(params["pad_b"])
        self.pad_l = int(params["pad_l"])
        self.pad_r = int(params["pad_r"])
        self.pad_mode = str(params["pad_mode"])
        # other parameters...



    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")

        #NCHW data format
        n=bottom[0].data.shape[0]
        c=bottom[0].data.shape[1]
        h=bottom[0].data.shape[2]
        w=bottom[0].data.shape[3]
        #top[0].reshape(n,c,h,w)
        #pad_dim = ((0,0),(0,0),(self.pad_t,self.pad_b),(self.pad_l,self.pad_r)) #2D padddings
        #new_shape=(np.pad(bottom[0].data, pad_dim, self.pad_mode)).shape
        top[0].reshape(n, c, h+self.pad_t+self.pad_b, w+self.pad_l+self.pad_r)


    def forward(self, bottom, top):
        pad_dim=((0,0),(0,0),(self.pad_t,self.pad_b),(self.pad_l,self.pad_r)) #2D padddings
        #top[0].data[...] = bottom[0].data[:]
        top[0].data[...] = np.pad(bottom[0].data, pad_dim, self.pad_mode)


    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
