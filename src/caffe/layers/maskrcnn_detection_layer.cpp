#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/maskrcnn_detection_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskRCNNDetectionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const MaskRCNNDetectionParameter &maskrcnn_detection_param =
      this->layer_param_.maskrcnn_detection_param();
  // maskrcnn_detection setup
  bbox_std_dev_.clear();
  std::copy(maskrcnn_detection_param.bbox_std_dev().begin(),
            maskrcnn_detection_param.bbox_std_dev().end(),
            std::back_inserter(bbox_std_dev_));
  batch_size_ = maskrcnn_detection_param.batch_size();
  detection_max_instances_ = maskrcnn_detection_param.detection_max_instances();
  detection_min_confidence_ =
      maskrcnn_detection_param.detection_min_confidence();
  detection_nms_threshold_ = maskrcnn_detection_param.detection_nms_threshold();

  // get num_class, N
  N_ = bottom[0]->shape(1);
  num_class_ = bottom[1]->shape(2);

  // check dimension of bottom
  CHECK_EQ(bottom[0]->num_axes(), 3)
      << "bottom[0] corresponds to rpn_rois, which has dimension: 3";
  CHECK_EQ(bottom[1]->num_axes(), 3)
      << "bottom[1] corresponds to mrcnn_class, which has dimension: 3";
  CHECK_EQ(bottom[2]->num_axes(), 4)
      << "bottom[2] corresponds to mrcnn_bbox, which has dimension: 4";
  CHECK_EQ(bottom[3]->num_axes(), 2)
      << "bottom[3] corresponds to input_image_meta, which has dimension: 2";
  CHECK_EQ(bottom[0]->shape(2), 4)
      << "bottom[0] should have shape: [batch_size, POST_NMS_ROIS_INFERENCE, "
         "4]";
  CHECK_EQ(bottom[1]->shape(2), bottom[2]->shape(2))
      << "bottom[1] and bottom[2] should have same shape in axis 2";
  CHECK_EQ(bottom[2]->shape(3), 4) << "bottom[2] should have shape 4 in axis 3";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
      << "bottom[0] and bottom[1] should have same shape in axis 1";
  CHECK_EQ(bottom[1]->shape(1), bottom[2]->shape(1))
      << "bottom[1] and bottom[2] should have same shape in axis 1";
}

template <typename Dtype>
void MaskRCNNDetectionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = {batch_size_, detection_max_instances_, 6};
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int>
MaskRCNNDetectionLayer<Dtype>::tf_nms(vector<vector<Dtype>> &pred_boxes,
                                      vector<int> &order, int topk,
                                      float iou_threshold) const {
  vector<int> class_keep;
  while (order.size() > 0) {
    int idx = order[0];
    class_keep.push_back(idx);
    vector<Dtype> inter(order.size() - 1);
    vector<Dtype> area_sum(order.size() - 1);
    for (int i = 0; i < order.size() - 1; ++i) {
      Dtype x1 =
          std::max<Dtype>(pred_boxes[idx][0], pred_boxes[order[i + 1]][0]);
      Dtype x2 =
          std::min<Dtype>(pred_boxes[idx][2], pred_boxes[order[i + 1]][2]);
      Dtype y1 =
          std::max<Dtype>(pred_boxes[idx][1], pred_boxes[order[i + 1]][1]);
      Dtype y2 =
          std::min<Dtype>(pred_boxes[idx][3], pred_boxes[order[i + 1]][3]);
      Dtype w = std::max<Dtype>(Dtype(0), (x2 - x1));
      Dtype h = std::max<Dtype>(Dtype(0), (y2 - y1));
      inter[i] = w * h;
      area_sum[i] =
          ((pred_boxes[idx][2] - pred_boxes[idx][0]) *
           (pred_boxes[idx][3] - pred_boxes[idx][1])) +
          ((pred_boxes[order[i + 1]][2] - pred_boxes[order[i + 1]][0]) *
           (pred_boxes[order[i + 1]][3] - pred_boxes[order[i + 1]][1]));
    }

    vector<int> left;
    for (int j = 0; j < order.size() - 1; ++j) {
      auto ratio =
          (area_sum[j] == 0) ? 0 : (inter[j] / (area_sum[j] - inter[j]));
      if (ratio < iou_threshold) {
        left.push_back(j + 1);
      }
    }
    if (left.size() == 0) {
      break;
    }
    for (int k = 0; k < left.size(); ++k) {
      order[k] = order[left[k]];
    }
    order.resize(left.size());
  }
  // while ends
  if (class_keep.size() > topk) {
    class_keep.resize(topk);
  }
  return class_keep;
}

template <typename Dtype>
void MaskRCNNDetectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  const Dtype *rois_data = bottom[0]->cpu_data();
  const Dtype *class_data = bottom[1]->cpu_data();
  const Dtype *bbox_data = bottom[2]->cpu_data();
  const Dtype *image_data = bottom[3]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  // get width and height from image_meta, which should be the same for all
  // batch data
  const float height = image_data[4];
  const float width = image_data[5];

  // input: [rois, mrcnn_class, mrcnn_bbox, window]  output: detections_batch
  for (int b = 0; b < batch_size_; ++b) {
    // get window from norm_boxes_graph(boxes, shape)
    vector<Dtype> window(4);
    window[0] = image_data[b * 4 + 7] / (height - 1.0);
    window[1] = image_data[b * 4 + 8] / (width - 1.0);
    window[2] = (image_data[b * 4 + 9] - 1.0) / (height - 1.0);
    window[3] = (image_data[b * 4 + 10] - 1.0) / (width - 1.0);

    vector<Dtype> class_scores(N_);
    vector<int> class_ids(N_);
    vector<Dtype> delta_specific(N_ * 4);
    vector<Dtype> refined_rois(N_ * 4);
    vector<int> keep;

    for (int i = 0; i < N_; ++i) {
      // probs = class_data[b][:][:]
      vector<Dtype> part_class_data(num_class_, 0);
      for (int p = 0; p < num_class_; ++p) {
        part_class_data[p] =
            class_data[b * N_ * num_class_ + i * num_class_ + p];
      }

      class_scores[i] =
          *(std::max_element(part_class_data.begin(), part_class_data.end()));
      class_ids[i] = std::distance(
          part_class_data.begin(),
          std::max_element(part_class_data.begin(), part_class_data.end()));

      // bbox_data[b][i][class_ids[i]][0] ~ bbox_data[b][i][class_ids[i]][3]
      // delta_specific = delta_specific*bbox_std_dev_
      int b2 = b * bottom[2]->count(1) + i * bottom[2]->count(2) +
               class_ids[i] * bottom[2]->count(3);
      delta_specific[i * 4] = bbox_data[b2] * bbox_std_dev_[0];
      delta_specific[i * 4 + 1] = bbox_data[b2 + 1] * bbox_std_dev_[1];
      delta_specific[i * 4 + 2] = bbox_data[b2 + 2] * bbox_std_dev_[2];
      delta_specific[i * 4 + 3] = bbox_data[b2 + 3] * bbox_std_dev_[3];
      // get rois from bottom[0] rois_data
      int b3 = b * bottom[0]->count(1) + i * bottom[0]->count(2);
      auto center_y = rois_data[b3] + (0.5 + delta_specific[i * 4]) *
                                          (rois_data[b3 + 2] - rois_data[b3]);
      auto center_x =
          rois_data[b3 + 1] + (0.5 + delta_specific[i * 4 + 1]) *
                                  (rois_data[b3 + 3] - rois_data[b3 + 1]);
      auto height = (rois_data[b3 + 2] - rois_data[b3]) *
                    std::exp(delta_specific[i * 4 + 2]);
      auto width = (rois_data[b3 + 3] - rois_data[b3 + 1]) *
                   std::exp(delta_specific[i * 4 + 3]);
      refined_rois[i * 4] = center_y - 0.5 * height;
      refined_rois[i * 4 + 1] = center_x - 0.5 * width;
      refined_rois[i * 4 + 2] = center_y + 0.5 * height;
      refined_rois[i * 4 + 3] = center_x + 0.5 * width;
      refined_rois[i * 4] =
          std::max(std::min(refined_rois[i * 4], window[2]), window[0]);
      refined_rois[i * 4 + 1] =
          std::max(std::min(refined_rois[i * 4 + 1], window[3]), window[1]);
      refined_rois[i * 4 + 2] =
          std::max(std::min(refined_rois[i * 4 + 2], window[2]), window[0]);
      refined_rois[i * 4 + 3] =
          std::max(std::min(refined_rois[i * 4 + 3], window[3]), window[1]);

      if ((class_ids[i] > 0) && (detection_min_confidence_ > 0)) {
        if (class_scores[i] > detection_min_confidence_) {
          keep.push_back(i);
        }
      }
    }
    vector<int> pre_nms_class_ids(keep.size());
    vector<Dtype> pre_nms_scores(keep.size());
    vector<Dtype> pre_nms_rois(keep.size() * 4);
    vector<int> unique_pre_nms_class_ids;
    for (int k = 0; k < keep.size(); ++k) {
      pre_nms_class_ids[k] = class_ids[keep[k]];
      pre_nms_scores[k] = class_scores[keep[k]];
      pre_nms_rois[k * 4] = refined_rois[keep[k] * 4];
      pre_nms_rois[k * 4 + 1] = refined_rois[keep[k] * 4 + 1];
      pre_nms_rois[k * 4 + 2] = refined_rois[keep[k] * 4 + 2];
      pre_nms_rois[k * 4 + 3] = refined_rois[keep[k] * 4 + 3];
      if (std::find(unique_pre_nms_class_ids.begin(),
                    unique_pre_nms_class_ids.end(),
                    pre_nms_class_ids[k]) == unique_pre_nms_class_ids.end()) {
        unique_pre_nms_class_ids.push_back(pre_nms_class_ids[k]);
      }
    }

    vector<int> nms_keep(
        unique_pre_nms_class_ids.size() * detection_max_instances_, -1);
    for (int d = 0; d < unique_pre_nms_class_ids.size(); ++d) {
      int id = unique_pre_nms_class_ids[d];
      vector<int> ixs;
      for (int i = 0; i < pre_nms_class_ids.size(); ++i) {
        if (pre_nms_class_ids[i] == id) {
          ixs.push_back(i);
        }
      }
      // get nms input: boxes = pre_nms_rois[ixs]
      vector<vector<Dtype>> nms_boxes(
          ixs.size(),
          vector<Dtype>(4)); // copy boxes value, shape=[ixs.size(), 4]
      vector<Dtype> nms_scores(ixs.size());
      vector<int> nms_order(ixs.size());
      for (int i = 0; i < ixs.size(); ++i) {
        nms_scores[i] = pre_nms_scores[ixs[i]];
        nms_order[i] = i;
        for (int j = 0; j < 4; ++j) {
          nms_boxes[i][j] = pre_nms_rois[ixs[i] * 4 + j];
        }
      }
      // get nms_order = np.argsort(-pre_nms_scores[ixs])
      std::sort(nms_order.begin(), nms_order.end(),
                [&](int i, int j) { return nms_scores[i] > nms_scores[j]; });
      // get nms output
      vector<int> class_keep =
          tf_nms(nms_boxes, nms_order, detection_max_instances_,
                 detection_nms_threshold_);

      for (int i = 0; i < class_keep.size(); ++i) {
        class_keep[i] = ixs[class_keep[i]];
      }
      for (int i = 0; i < class_keep.size(); ++i) {
        class_keep[i] = keep[class_keep[i]];
        nms_keep[d * detection_max_instances_ + i] = class_keep[i];
      }
    }

    vector<int> keep2;
    for (int i = 0; i < nms_keep.size(); ++i) {
      if (nms_keep[i] > -1) {
        for (int j = 0; j < keep.size(); ++j) {
          if (nms_keep[i] == keep[j]) {
            keep2.push_back(nms_keep[i]);
          }
        }
      }
    }
    vector<Dtype> class_scores_keep(keep2.size());
    vector<int> top_ids(class_scores_keep.size());
    for (int i = 0; i < keep2.size(); ++i) {
      class_scores_keep[i] = class_scores[keep2[i]];
      top_ids[i] = i;
    }
    int num_keep = (keep2.size() > detection_max_instances_)
                       ? detection_max_instances_
                       : keep2.size();
    // get tf.nn.top_k output
    std::sort(top_ids.begin(), top_ids.end(), [&](int i, int j) {
      return class_scores_keep[i] > class_scores_keep[j];
    });
    vector<int> keep3;
    for (int i = 0; i < num_keep; ++i) {
      keep3.push_back(keep2[top_ids[i]]);
    }

    // set initialized top_data as 0 to replace tf.pad
    caffe_set(top[0]->count(), Dtype(0.), top_data);
    for (int i = 0; i < keep3.size(); ++i) {
      int kidx = keep3[i];
      int top_idx = b * top[0]->count(1) + i * top[0]->count(2);
      // tf.gather(refined_rois, keep)
      top_data[top_idx] = refined_rois[kidx * 4];
      top_data[top_idx + 1] = refined_rois[kidx * 4 + 1];
      top_data[top_idx + 2] = refined_rois[kidx * 4 + 2];
      top_data[top_idx + 3] = refined_rois[kidx * 4 + 3];
      // tf.gather(class_ids, keep)
      top_data[top_idx + 4] = class_ids[kidx];
      // tf.gather(class_scores, keep)
      top_data[top_idx + 5] = class_scores[kidx];
    }
  }
}

INSTANTIATE_CLASS(MaskRCNNDetectionLayer);
REGISTER_LAYER_CLASS(MaskRCNNDetection);

} // namespace caffe
