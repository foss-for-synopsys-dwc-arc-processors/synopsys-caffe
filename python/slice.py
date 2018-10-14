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
        ndim = bottom[0].data.ndim
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
        length_diff = ndim - len(begin)
        if length_diff > 0:
            begin.extend(length_diff*[0])
            end.extend(shape[-length_diff:])
            strides.extend([1]*length_diff)
        elif length_diff < 0:
            raise Exception("The length of begin exceeds ndim")
        self.begin_mask = params.get("begin_mask", 0)
        self.end_mask = params.get("end_mask", 0)
        self.ellipsis_mask = params.get("ellipsis_mask", 0)
        self.new_axis_mask = params.get("new_axis_mask", 0)
        self.shrink_axis_mask = params.get("shrink_axis_mask", 0)

        if self.new_axis_mask:
            raise NotImplementedError("new_axis_mask is not implemented")
        self.shrink_axis = []
        for i, d in enumerate(shape):
            if self.begin_mask & 1 << i:
                begin[i] = 0
            if self.end_mask & 1 << i:
                end[i] = shape[i]
            if self.ellipsis_mask & 1 << i:
                begin[i] = 0
                end[i] = shape[i]
                strides[i] = 1
            if self.shrink_axis_mask & 1 << i:
                end[i] = begin[i] + 1
                strides[i] = 1
                self.shrink_axis.append(i)
        self.begin = begin
        self.end = end
        self.strides = strides

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        ndim = bottom[0].data.ndim
        num = [0]*ndim
        for i in range(ndim):
            if self.strides[i] == 0:
                raise Exception("Strides should never be equal to 0!")
            else:
                num[i] = abs(self.end[i]-self.begin[i])/abs(self.strides[i])
        for i in reversed(self.shrink_axis):
            num.pop(i)
        top[0].reshape(*num)

    def forward(self, bottom, top):
        idx = tuple(slice(s[0], s[1], s[2])
                    for s in zip(self.begin, self.end, self.strides))
        # if self.shrink_axis_mask:
        #     top[0].data[...] = np.squeeze(
        #         bottom[0].data[idx], axis=tuple(self.shrink_axis))
        # else:
        top[0].data[...] = bottom[0].data[idx]

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
