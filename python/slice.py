import caffe
import numpy as np


class Slice(caffe.Layer):
    """
    Get a tensor's slice: implementation of the tf.slice().
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        # bottom[1] is begin
        # bottom[2] is size
        if len(bottom) != 3:
            raise Exception("Please input three Tensors!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        assert bottom[1].data.ndim <= 1
        assert bottom[2].data.ndim <= 1

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        top[0].reshape(*bottom[2].data.shape)

    def forward(self, bottom, top):
        idx = tuple(slice(int(s[0]), int(s[0]+s[1]))
                    for s in zip(bottom[1].data, bottom[2].data))
        top[0].data[...] = bottom[0].data[idx]

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]


class StridedSlice(caffe.Layer):
    """
    Get a tensor's slicing: implementation of the tf.strided_slice().
    """
    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        shape = bottom[0].data.shape

        params = eval(self.param_str)
        begin = params["begin"]
        end = params["end"]
        strides = params.get("strides")
        if not strides:
            strides = [1]*len(begin)
        if len(begin) != len(end) or len(end) != len(strides):
            raise ValueError(
                "begin, end and strides should be the same length")
        begin_mask = params.get("begin_mask", 0)
        end_mask = params.get("end_mask", 0)
        ellipsis_mask = params.get("ellipsis_mask", 0)
        new_axis_mask = params.get("new_axis_mask", 0)
        shrink_axis_mask = params.get("shrink_axis_mask", 0)
        for i in range(len(begin)):
            if begin_mask & 1 << i:
                begin[i] = 0
            if end_mask & 1 << i:
                end[i] = shape[i]
            if shrink_axis_mask & 1 << i:
                end[i] = begin[i] + 1
                strides[i] = 1
        slices = [slice(*i) for i in zip(begin, end, strides)]
        ellipsis_axis = None

        for i in range(len(slices)):
            if ellipsis_mask & 1 << i:
                slices[i] = Ellipsis
                ellipsis_axis = i
            if new_axis_mask & 1 << i:
                slices[i] = np.newaxis
        self.slices = tuple(slices)
        # shape for reshape
        shape = list(bottom[0].data[self.slices].shape)
        ellipsis_expansion = len(shape) - len(begin)
        # sqeeze the shape in the positions in shrink_axis_mask
        for i in range(len(shape)):
            if shrink_axis_mask & 1 << i:
                if i < ellipsis_axis:
                    del shape[i]
                else:
                    del shape[i+ellipsis_expansion]
        self.shape = shape

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        top[0].reshape(*self.shape)

    def forward(self, bottom, top):
        top[0].data[:] = bottom[0].data[self.slices][:]

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
