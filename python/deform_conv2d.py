import caffe
import numpy as np
import torch
import torchvision.ops

class Deform_Conv2D(caffe.Layer):
    """
    Implemention of pytorch deform_conv2d method.
    Refer to https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html

    Used for DBNet deploy https://github.com/MhLiao/DB/blob/master/assets/ops/dcn/functions/deform_conv.py#L111
    Input:
        input,
        offset,
        mask,
        weight,
        # bias=None,  #
    Param:
        stride=1,
        padding=1, #
        dilation=1,
        groups=1,  #
        deformable_groups=1  #
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 4:
            raise Exception("Only supporting input 4 Tensors now!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        
        # d = eval(self.param_str)
        d = dict()
        self.stride = d.get("stride", 1)
        self.padding = d.get("padding", 1)
        self.dilation = d.get("dilation", 1)

    def reshape(self, bottom, top):
        # check input dimensions
        #if bottom[0].count == 0:
        #    raise Exception("Input must not be empty!")
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        input = bottom[0].data
        offset = bottom[1].data
        mask = bottom[2].data
        weight = bottom[3].data
        # bias  #
        x = torchvision.ops.deform_conv2d(
            input=torch.from_numpy(input),
            weight=torch.from_numpy(weight),
            # bias=torch.from_numpy(bias),
            offset=torch.from_numpy(offset),
            mask=torch.from_numpy(mask),
            stride=int(self.stride),
            padding=int(self.padding),
            dilation=int(self.dilation),
        )
        top[0].data[...] = x.detach().cpu().numpy()

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
