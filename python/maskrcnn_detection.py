import caffe
import numpy as np


class MaskRCNN_Detection(caffe.Layer):
    """
    Rewrite PyFunc (mrcnn_detection) in Mask RCNN
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 4:
            raise Exception("Only input 4 Tensors at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
            
        params = eval(self.param_str)
        self.BATCH_SIZE = params["batch_size"]
        self.IMAGES_PER_GPU = params["images_per_gpu"]
        self.DETECTION_MAX_INSTANCES = params["detection_max_instances"]
        self.DETECTION_MIN_CONFIDENCE = params["detection_min_confidence"]
        self.DETECTION_NMS_THRESHOLD = params["detection_nms_threshold"]
        self.BBOX_STD_DEV = params["bbox_std_dev"]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        top[0].reshape(self.BATCH_SIZE, self.DETECTION_MAX_INSTANCES, 6)

    def gather_nd(self, params, indices):
        i_shape = list(indices.shape)
        p_shape = list(params.shape)
        o_shape = i_shape[:-1] + p_shape[i_shape[-1]:]
        count = int(np.size(indices) / i_shape[-1])
        indices_2d = indices.reshape(count, i_shape[-1])
        output = np.expand_dims(params[tuple(indices_2d[0])], axis=0)
        for i in range(1,count):
            part = np.expand_dims(params[tuple(indices_2d[i])], axis=0)
            output = np.concatenate([output, part], axis=0)
        output.reshape(o_shape)    

        return output

    def nms(self, bounding_boxes, score, topk, threshold):
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

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
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

    def refine_detections_graph(self, rois, probs, deltas, window):
        """Refine classified proposals and filter overlaps and return final
        detections.
        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
                that contains the image excluding the padding.
        Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
            coordinates are normalized.
        """
        # Class IDs per ROI
        class_ids = np.argmax(probs, axis=1)
        # Class probability of the top class of each ROI
        indices = np.stack([np.arange(probs.shape[0]), class_ids], axis=1)
        class_scores = self.gather_nd(probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = self.gather_nd(deltas, indices)
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = self.apply_box_deltas_graph(
            rois, deltas_specific * self.BBOX_STD_DEV)
        # Clip boxes to image window
        refined_rois = self.clip_boxes_graph(refined_rois, window)

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = np.where(class_ids > 0)[0]
        # Filter out low confidence boxes
        if self.DETECTION_MIN_CONFIDENCE:
            conf_keep = np.where(class_scores >= self.DETECTION_MIN_CONFIDENCE)[0]
            keep = np.array(sorted(list(set(keep).intersection(set(conf_keep)))))

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = class_ids[keep]
        pre_nms_scores = class_scores[keep]
        pre_nms_rois = refined_rois[keep]
        unique_pre_nms_class_ids, idx = np.unique(pre_nms_class_ids, return_index=True)
        unique_pre_nms_class_ids = unique_pre_nms_class_ids[np.argsort(idx)]

        def nms_keep_map(class_id):
            """Apply Non-Maximum Suppression on ROIs of the given class."""
            # Indices of ROIs of the given class
            ixs = np.where(pre_nms_class_ids == class_id)[0]
            # Apply NMS
            class_keep = self.nms(
                    pre_nms_rois[ixs],
                    pre_nms_scores[ixs],
                    topk=self.DETECTION_MAX_INSTANCES,
                    threshold=self.DETECTION_NMS_THRESHOLD)
            # Map indices
            class_keep = ixs[class_keep]
            class_keep = keep[class_keep]
            # Pad with -1 so returned tensors have the same shape
            gap = self.DETECTION_MAX_INSTANCES - class_keep.shape[0]
            class_keep = np.pad(class_keep, [(0, gap)],
                                mode='constant', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            class_keep = class_keep.reshape([self.DETECTION_MAX_INSTANCES])
            return class_keep

        # 2. Map over class IDs
        nms_keep = np.array(list(map(nms_keep_map, unique_pre_nms_class_ids)))
        # 3. Merge results into one list, and remove -1 padding
        nms_keep = nms_keep.reshape([-1])
        nms_keep = nms_keep[np.where(nms_keep > -1)[0]]
        # 4. Compute intersection between keep and nms_keep
        keep = np.array(sorted(list(set(keep).intersection(set(nms_keep)))))

        # Keep top detections
        roi_count = self.DETECTION_MAX_INSTANCES
        class_scores_keep = class_scores[keep]
        num_keep = np.minimum(class_scores_keep.shape[0], roi_count)
        top_ids = np.argsort(-class_scores_keep)[...,:num_keep]
        keep = keep[top_ids]

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are normalized.
        detections = np.concatenate([
            refined_rois[keep],
            class_ids[keep][..., np.newaxis],
            class_scores[keep][..., np.newaxis]
            ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = self.DETECTION_MAX_INSTANCES - detections.shape[0]
        detections = np.pad(detections, [(0, gap), (0, 0)], "constant")    
        return detections


    def norm_boxes_graph(self, boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels
        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.
        Returns:
            [..., (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = np.split(shape, 2)
        scale = np.concatenate([h, w, h, w], axis=-1) - 1.0
        shift = np.array([0., 0., 1., 1.])
        output = np.divide(boxes - shift, scale)
        return output

    def batch_slice(self, inputs, graph_fn, batch_size):
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

    def mrcnn_detection(self, rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta):
        rois = rpn_rois
        mrcnn_class = mrcnn_class
        mrcnn_bbox = mrcnn_bbox
        image_meta = input_image_meta

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.

        image_shape = image_meta[:, 4:7][0]
        window = self.norm_boxes_graph(image_meta[:, 7:11], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = self.batch_slice(
                [rois, mrcnn_class, mrcnn_bbox, window],
                lambda x, y, w, z: self.refine_detections_graph(x, y, w, z), self.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        detections = detections_batch.reshape([self.BATCH_SIZE, self.DETECTION_MAX_INSTANCES, 6])   # [batch_size,100,6]  

        return detections

    def forward(self, bottom, top):
        rpn_rois = bottom[0].data
        mrcnn_class = bottom[1].data
        mrcnn_bbox = bottom[2].data
        input_image_meta = bottom[3].data

        detections_batch = self.mrcnn_detection(rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta)
        #print(detections_batch)
        top[0].data[...] = detections_batch

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
