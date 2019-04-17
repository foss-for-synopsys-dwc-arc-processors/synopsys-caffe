import caffe
import numpy as np


class Pre_Roi_Align(caffe.Layer):
    """
    Implement the previous part of ROI Align in Mask RCNN
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        self.height = 1024
        self.width =1024

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        shape = list(bottom[0].data.shape)
        shape.pop(-1)
        shape = [int(i) for i in shape]
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        boxes = bottom[0].data

        y1, x1, y2, x2 = np.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        image_area = self.height * self.width
        roi_level = np.log2(np.sqrt(h * w) / (224.0 / np.sqrt(image_area)))
        roi_level = np.minimum(5, np.maximum(2, 4 + np.round(roi_level)))
        roi_level = np.squeeze(roi_level, 2)

        top[0].data[...] = roi_level

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
