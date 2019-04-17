import caffe
import numpy as np


class Apply_Box_Deltas(caffe.Layer):
    """
    Implement part of RPN in Mask RCNN
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 2:
            raise Exception("Only input 2 Tensors at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        self.height = 1024
        self.width =1024

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = bottom[0].data.shape
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        boxes = bottom[0].data
        deltas = bottom[1].data

        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= np.exp(deltas[:, 2])
        width *= np.exp(deltas[:, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width

        wy1 = 0
        wx1 = 0
        wy2 = self.height
        wx2 = self.width
        y1 = np.maximum(np.minimum(y1, wy2), wy1)
        x1 = np.maximum(np.minimum(x1, wx2), wx1)
        y2 = np.maximum(np.minimum(y2, wy2), wy1)
        x2 = np.maximum(np.minimum(x2, wx2), wx1)
        result = np.stack([y1, x1, y2, x2], axis=1)

        top[0].data[...] = result

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
