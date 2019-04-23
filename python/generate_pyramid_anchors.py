import caffe
import numpy as np
import math


class Generate_Pyramid_Anchors(caffe.Layer):
    """For Mask RCNN Proposal layer
    Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")

        self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
        self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
        # Anchor stride
        # If 1 then anchors are created for each cell in the backbone feature map.
        # If 2, then anchors are created for every other cell, and so on.
        self.RPN_ANCHOR_STRIDE = 1

        self.HEIGHT = 1024
        self.WIDTH =1024

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.HEIGHT / stride)),
              int(math.ceil(self.WIDTH / stride))]
             for stride in self.BACKBONE_STRIDES])

    def reshape(self, bottom, top):
        shape = self.BACKBONE_SHAPES / self.RPN_ANCHOR_STRIDE
        shape = np.dot(shape[:, 0], shape[:, 1])
        shape = shape * len(self.RPN_ANCHOR_RATIOS)
        shape = [shape, 4]

        top[0].reshape(*shape)


    def generate_anchors(self, scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    def forward(self, bottom, top):
        anchors = []
        for i in range(len(self.RPN_ANCHOR_SCALES)):
            anchors.append(self.generate_anchors(self.RPN_ANCHOR_SCALES[i], self.RPN_ANCHOR_RATIOS, self.BACKBONE_SHAPES[i],
                                            self.BACKBONE_STRIDES[i], self.RPN_ANCHOR_STRIDE))

        top[0].data[...] = np.concatenate(anchors, axis=0)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
