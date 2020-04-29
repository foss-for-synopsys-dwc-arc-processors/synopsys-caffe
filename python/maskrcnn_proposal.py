import caffe
import numpy as np


class MaskRCNN_Proposal(caffe.Layer):
    """
    Rewrite ProposalLayer in Mask RCNN """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 3:
            raise Exception("Only input 3 Tensors at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        try:
            params = eval(self.param_str)
            if params["height"] is not None:
                self.HEIGHT = int(params["height"])
            else:
                self.HEIGHT = 1024  # 1920
            if params["width"] is not None:
                self.WIDTH = int(params["width"])
            else:
                self.WIDTH = 1024  # 1920
        except Exception as ex:
            print("No params set, use default input dim instead:")
            self.HEIGHT = 1024  # 1920
            self.WIDTH = 1024  # 1920
            print("Height:", self.HEIGHT, " Width:", self.WIDTH)

        params = eval(self.param_str)
        self.BATCH_SIZE = np.array(params["batch_size"])
        self.IMAGES_PER_GPU = np.array(params["images_per_gpu"])
        self.RPN_BBOX_STD_DEV = np.array(params["rpn_bbox_std_dev"])
        self.PRE_NMS_LIMIT = np.array(params["pre_nms_limit"])
        self.RPN_NMS_THRESHOLD = np.array(params["rpn_nms_threshold"])
        self.POST_NMS_ROIS_INFERENCE = np.array(params["post_nms_rois_inference"])

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        top[0].reshape(self.BATCH_SIZE, 1000, 4)

    def batch_slice(self, inputs, graph_fn, batch_size, names=None):
        """Splits inputs into slices and feeds each slice to a copy of the given
        computation graph and then combines the results. It allows you to run a
        graph on a batch of inputs even if the graph is written to support one
        instance only.
        inputs: list of tensors. All must have the same first dimension length
        graph_fn: A function that returns a TF tensor that's part of a graph.
        batch_size: number of slices to divide the data into.
        names: If provided, assigns names to the resulting tensors.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        outputs = []
        for i in range(batch_size):
            inputs_slice = [x[i] for x in inputs]
            output_slice = graph_fn(*inputs_slice)
            if not isinstance(output_slice, (tuple, list)):
                output_slice = [output_slice]
            outputs.append(output_slice)
            # Change outputs from a list of slices where each is
            # a list of outputs to a list of outputs and each has
            # a list of slices
        outputs = list(zip(*outputs))

        result = [np.stack(o, axis=0) for o in outputs]
        if len(result) == 1:
            result = result[0]

        return result

    def apply_box_deltas_graph(self, boxes, deltas):
        """Applies the given deltas to the given boxes.
        boxes: [N, (y1, x1, y2, x2)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        """
        # Convert to y, x, h, w
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
        result = np.stack([y1, x1, y2, x2], axis=1)
        return result

    def clip_boxes_graph(self, boxes, window):
        """
        boxes: [N, (y1, x1, y2, x2)]
        window: [4] in the form y1, x1, y2, x2
        """
        # Split
        wy1, wx1, wy2, wx2 = np.split(window, 4)
        y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
        # Clip
        y1 = np.maximum(np.minimum(y1, wy2), wy1)
        x1 = np.maximum(np.minimum(x1, wx2), wx1)
        y2 = np.maximum(np.minimum(y2, wy2), wy1)
        x2 = np.maximum(np.minimum(x2, wx2), wx1)
        clipped = np.concatenate([y1, x1, y2, x2], axis=1)
        clipped = clipped.reshape((clipped.shape[0], 4))
        return clipped

    def image_nms(self, bounding_boxes, score, topk, threshold):
        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], []
        # coordinates of bounding boxes
        start_x = bounding_boxes[:, 0]
        start_y = bounding_boxes[:, 1]
        end_x = bounding_boxes[:, 2]
        end_y = bounding_boxes[:, 3]

        # Compute areas of bounding boxes
        areas = (end_x - start_x) * (end_y - start_y)

        # Sort by confidence score of bounding boxes
        order = score.argsort()

        idx = []
        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]
            idx.append(index)

            # Compute ordinates of intersection - over - union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection - over - union
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
            intersection = w * h

            # Compute the ratio between intersection and union
            area_sum = areas[index] + areas[order[:-1]]
            l = len(areas[order[:-1]])
            ratio = np.zeros(l)
            for i in range(l):
                if area_sum[i] == 0:
                    ratio[i] = 0
                else:
                    ratio[i] = intersection[i] / (area_sum[i] - intersection[i])
            left = np.where(np.array(ratio) < threshold)
            order = order[left]

        return idx[:topk]

    def mrcnn_proposal(self, rpn_class, rpn_bbox, anchors):
        # Box Scores.Use the foreground class confidence.[Batch, num_rois, 1]
        scores = rpn_class[:, :, 1]
        # Box deltas[batch, num_rois, 4]
        deltas = rpn_bbox
        deltas = deltas * np.reshape(self.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = anchors
        proposal_count = self.POST_NMS_ROIS_INFERENCE
        nms_threshold = self.RPN_NMS_THRESHOLD

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = np.minimum(self.PRE_NMS_LIMIT, np.shape(anchors)[1])
        ix = np.argsort(-scores)[..., :pre_nms_limit]
        scores = self.batch_slice(
            [scores, ix],
            lambda x, y: x[y],
            self.IMAGES_PER_GPU)
        deltas = self.batch_slice(
            [deltas, ix],
            lambda x, y: x[y],
            self.IMAGES_PER_GPU)
        pre_nms_anchors = self.batch_slice(
            [anchors, ix],
            lambda a, x: a[x],
            self.IMAGES_PER_GPU)

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = self.batch_slice(
            [pre_nms_anchors, deltas],
            lambda x, y: self.apply_box_deltas_graph(x, y),
            self.IMAGES_PER_GPU)

        # Clip to image boundaries.Since we're in normalized coordinates,
        # clip to 0..1 range.[batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = self.batch_slice(
            boxes,
            lambda x: self.clip_boxes_graph(x, window),
            self.IMAGES_PER_GPU)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non - max suppression
        def nms(boxes, scores):
            indices = self.image_nms(
                boxes, scores, proposal_count,
                nms_threshold)
            proposals = boxes[indices]
            # Pad if needed
            padding = np.maximum(proposal_count - np.shape(proposals)[0], 0)
            proposals = np.pad(
                proposals, [(0, padding), (0, 0)], mode='constant')
            return proposals
        proposals = self.batch_slice([boxes, scores], nms, self.IMAGES_PER_GPU)
        return proposals

    def forward(self, bottom, top):
        rpn_class = bottom[0].data
        rpn_bbox = bottom[1].data
        anchors = bottom[2].data
        rpn_rois = self.mrcnn_proposal(rpn_class, rpn_bbox, anchors)

        top[0].data[...] = rpn_rois

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
