import caffe
import numpy as np


class MaskRCNN_Detection(caffe.Layer):
    """
    Rewrite PyFunc (mrcnn_detection) in Mask RCNN
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 4:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
            raise Exception("Only output one Tensor at a time!")
        try:
            params = eval(self.param_str)
            if params["height"] != None:
                self.HEIGHT = int(params["height"])
            else:
                self.HEIGHT = 1024  # 1920
            if params["width"] != None:
                self.WIDTH = int(params["width"])
            else:
                self.WIDTH = 1024  # 1920
        except Exception as ex:
            print "No params set, use default input dim instead:"
            self.HEIGHT = 1024  # 1920
            self.WIDTH = 1024  # 1920
            print "Height:", self.HEIGHT, " Width:", self.WIDTH

        self.BATCH_SIZE = 1
        self.DETECTION_MAX_INSTANCES = 100
        self.DETECTION_MIN_CONFIDENCE = 0.7
        self.DETECTION_NMS_THRESHOLD = 0.3
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
        top[0].reshape(1, 100, 6)

    def apply_box_deltas(self, boxes, deltas):
        """Applies the given deltas to the given boxes.
        boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
        deltas: [N, (dy, dx, log(dh), log(dw))]
        """
        boxes = boxes.astype(np.float32)
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
        return np.stack([y1, x1, y2, x2], axis=1)

    def compute_iou(self, box, boxes, box_area, boxes_area):
        """Calculates IoU of the given box with the array of the given boxes.
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

        Note: the areas are passed in rather than calculated here for
              efficency. Calculate once in the caller to avoid duplicate work.
        """
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou

    def non_max_suppression(self, boxes, scores, threshold):
        """Performs non-maximum supression and returns indicies of kept boxes.
        boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
        scores: 1-D array of box scores.
        threshold: Float. IoU threshold to use for filtering.
        """
        assert boxes.shape[0] > 0
        if boxes.dtype.kind != "f":
            boxes = boxes.astype(np.float32)

        # Compute box areas
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
        area = (y2 - y1) * (x2 - x1)

        # Get indicies of boxes sorted by scores (highest first)
        ixs = scores.argsort()[::-1]

        pick = []
        while len(ixs) > 0:
            # Pick top box and add its index to the list
            i = ixs[0]
            pick.append(i)
            # Compute IoU of the picked box with the rest
            iou = self.compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
            # Identify boxes with IoU over the threshold. This
            # returns indicies into ixs[1:], so add 1 to get
            # indicies into ixs.
            remove_ixs = np.where(iou > threshold)[0] + 1
            # Remove indicies of the picked and overlapped boxes.
            ixs = np.delete(ixs, remove_ixs)
            ixs = np.delete(ixs, 0)
        return np.array(pick, dtype=np.int32)

    def clip_to_window(self, window, boxes):
        """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
        """
        boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
        boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
        boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
        boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
        return boxes

    def parse_image_meta(self, meta):
        """Parses an image info Numpy array to its components.
        See compose_image_meta() for more details.
        """
        image_id = meta[:, 0]
        image_shape = meta[:, 1:4]
        window = meta[:, 4:8]  # (y1, x1, y2, x2) window of image in in pixels
        active_class_ids = meta[:, 8:]
        return image_id, image_shape, window, active_class_ids

    def refine_detections(self, rois, probs, deltas, window):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in image coordinates. The part of the image
                that contains the image excluding the padding.

        Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
        """
        # Class IDs per ROI
        class_ids = np.argmax(probs, axis=1)
        # Class probability of the top class of each ROI
        class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
        # Class-specific bounding box deltas
        deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = self.apply_box_deltas(
            rois, deltas_specific * self.BBOX_STD_DEV)
        # Convert coordiates to image domain
        # TODO: better to keep them normalized until later
        height = self.HEIGHT
        width = self.WIDTH
        refined_rois *= np.array([height, width, height, width])
        # Clip boxes to image window
        refined_rois = self.clip_to_window(window, refined_rois)
        # Round and cast to int since we're deadling with pixels now
        refined_rois = np.rint(refined_rois).astype(np.int32)
        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = np.where(class_ids > 0)[0]
        # Filter out low confidence boxes
        if self.DETECTION_MIN_CONFIDENCE:
            keep = np.intersect1d(
                keep, np.where(class_scores >= self.DETECTION_MIN_CONFIDENCE)[0])
        # Apply per-class NMS
        pre_nms_class_ids = class_ids[keep]
        pre_nms_scores = class_scores[keep]
        pre_nms_rois = refined_rois[keep]
        nms_keep = []
        for class_id in np.unique(pre_nms_class_ids):
            # Pick detections of this class
            ixs = np.where(pre_nms_class_ids == class_id)[0]
            # Apply NMS
            class_keep = self.non_max_suppression(
                pre_nms_rois[ixs], pre_nms_scores[ixs],
                self.DETECTION_NMS_THRESHOLD)
            # Map indicies
            class_keep = keep[ixs[class_keep]]
            nms_keep = np.union1d(nms_keep, class_keep)
        keep = np.intersect1d(keep, nms_keep).astype(np.int32)

        # Keep top detections
        roi_count = self.DETECTION_MAX_INSTANCES
        top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
        keep = keep[top_ids]

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are in image domain.
        result = np.hstack((refined_rois[keep],
                            class_ids[keep][..., np.newaxis],
                            class_scores[keep][..., np.newaxis]))
        return result

    def forward(self, bottom, top):
        rois = bottom[0].data
        mrcnn_class = bottom[1].data
        mrcnn_bbox = bottom[2].data
        image_meta = bottom[3].data

        detections_batch = []
        _, _, window, _ = self.parse_image_meta(image_meta)
        for b in range(self.BATCH_SIZE):
            detections = self.refine_detections(
                rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b])
            # Pad with zeros if detections < DETECTION_MAX_INSTANCES
            gap = self.DETECTION_MAX_INSTANCES - detections.shape[0]
            assert gap >= 0
            if gap > 0:
                detections = np.pad(
                    detections, [(0, gap), (0, 0)], 'constant', constant_values=0)
            detections_batch.append(detections)
        detections_batch = np.array(detections_batch).astype(np.float32)
        #print detections_batch
        top[0].data[...] = detections_batch

    def backward(self, top, propagate_down, bottom):
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]
