#include <algorithm>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include "caffe/layers/maskrcnn_proposal_layer.hpp"
using namespace std;
namespace caffe {

template <typename Dtype>
void MaskRCNNProposalLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  CHECK_EQ(bottom.size(), 3) << "Only input 3 Tensors at a time!";
  CHECK_EQ(top.size(), 1) << "Only output one Tensor at a time!";

 // int height = this->layer_param_.maskrcnn_proposal_param().height();
 // int width = this->layer_param_.maskrcnn_proposal_param().width();
  

  rpn_bbox_std_dev.clear();
  std::copy(
      this->layer_param_.maskrcnn_proposal_param().rpn_bbox_std_dev().begin(),
      this->layer_param_.maskrcnn_proposal_param().rpn_bbox_std_dev().end(),
      std::back_inserter(rpn_bbox_std_dev));

  batch_size = this->layer_param_.maskrcnn_proposal_param().batch_size();
  images_per_gpu =
      this->layer_param_.maskrcnn_proposal_param().images_per_gpu();
  // vector<float> RPN_BBOX_STD_DEV =
  // this->layer_param_.maskrcnn_proposal_param().rpn_bbox_std_dev();
  pre_nms_limit = this->layer_param_.maskrcnn_proposal_param().pre_nms_limit();
  rpn_nms_threshold =
      this->layer_param_.maskrcnn_proposal_param().rpn_nms_threshold();
  post_nms_rois_inference =
      this->layer_param_.maskrcnn_proposal_param().post_nms_rois_inference();

  num_rois = bottom[0]->shape(1);
}

template <typename Dtype>
void MaskRCNNProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {

  //      vector<int> top_shape = { batch_size,pre_nms_limit,4};
  vector<int> top_shape = {batch_size, post_nms_rois_inference, 4};
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
vector<vector<Dtype>> MaskRCNNProposalLayer<Dtype>::apply_box_deltas_graph(
    vector<vector<Dtype>> boxes, vector<vector<Dtype>> deltas) {

  int num = boxes.size();
  vector<vector<Dtype>> result(num, vector<Dtype>(4));

  Dtype height, width, center_y, center_x, y1, x1, y2, x2;
  for (int i = 0; i < num; i++) {
    height = boxes[i][2] - boxes[i][0];
    width = boxes[i][3] - boxes[i][1];
    center_y = boxes[i][0] + 0.5 * height;
    center_x = boxes[i][1] + 0.5 * width;

    center_y += deltas[i][0] * height;
    center_x += deltas[i][1] * width;

    height *= exp(deltas[i][2]);
    width *= exp(deltas[i][3]);

    y1 = center_y - 0.5 * height;
    x1 = center_x - 0.5 * width;
    y2 = y1 + height;
    x2 = x1 + width;
    result[i][0] = y1;
    result[i][1] = x1;
    result[i][2] = y2;
    result[i][3] = x2;
  }
  return result;
}

template <typename Dtype>
vector<vector<Dtype>>
MaskRCNNProposalLayer<Dtype>::clip_boxes_graph(vector<vector<Dtype>> boxes,
                                               vector<Dtype> window) {
  int num = boxes.size();
  vector<vector<Dtype>> clipped(num, vector<Dtype>(4));

  Dtype wy1, wx1, wy2, wx2, y1, x1, y2, x2;
  wy1 = window[0];
  wx1 = window[1];
  wy2 = window[2];
  wx2 = window[3];

  for (int i = 0; i < num; i++) {
    y1 = boxes[i][0];
    x1 = boxes[i][1];
    y2 = boxes[i][2];
    x2 = boxes[i][3];
    y1 = max(min(y1, wy2), wy1);
    x1 = max(min(x1, wx2), wx1);
    y2 = max(min(y2, wy2), wy1);
    x2 = max(min(x2, wx2), wx1);
    clipped[i][0] = y1;
    clipped[i][1] = x1;
    clipped[i][2] = y2;
    clipped[i][3] = x2;
  }
  return clipped;
}

template <typename T> vector<int> argsort(vector<T> &array) {
  const int array_len(array.size());
  std::vector<int> array_index(array_len, 0);
  for (int i = 0; i < array_len; ++i) {
    array_index[i] = i;
  }
  std::sort(
      array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] > array[pos2]); });
  return array_index;
}

template <typename Dtype>
vector<int>
MaskRCNNProposalLayer<Dtype>::image_nms(vector<vector<Dtype>> &pred_boxes,
                                        vector<int> &order, int topk,
                                        float iou_threshold) {
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

      //		cout<<"area_sum[i]" << area_sum[i]<<endl;
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
Dtype *MaskRCNNProposalLayer<Dtype>::maskrcnnproposal(const Dtype *rpn_class,
                                                      const Dtype *rpn_bbox,
                                                      const Dtype *anchors) {
  Dtype *rpn_rois =
      (Dtype *)malloc(batch_size * post_nms_rois_inference * 4 * sizeof(Dtype));

  vector<Dtype> scores(num_rois);
  vector<int> ix;

  int proposal_count = post_nms_rois_inference;
  float nms_threshold = rpn_nms_threshold;
  pre_nms_limit = std::min(pre_nms_limit, num_rois);

  for (int b = 0; b < batch_size; b++) {
    scores.clear();
    scores.resize(num_rois);
    for (int n = 0; n < num_rois; n++) {
      scores[n] = rpn_class[b * num_rois * 2 + n * 2 + 1];
    }
    ix = argsort(scores);

    ix.resize(pre_nms_limit);
    ix.shrink_to_fit();

    vector<vector<Dtype>> deltas_(ix.size(), vector<Dtype>(4));
    vector<vector<Dtype>> pre_nms_anchors_(ix.size(), vector<Dtype>(4));
    vector<Dtype> scores_(ix.size());
    vector<int> nms_order(ix.size());
    for (int i = 0; i < ix.size(); i++) {
      int idx = b * num_rois * 4 + ix.at(i) * 4;
      scores_[i] = scores[ix.at(i)];
      nms_order[i] = i;

      deltas_[i][0] = rpn_bbox[idx] * rpn_bbox_std_dev[0];
      deltas_[i][1] = rpn_bbox[idx + 1] * rpn_bbox_std_dev[1];
      deltas_[i][2] = rpn_bbox[idx + 2] * rpn_bbox_std_dev[2];
      deltas_[i][3] = rpn_bbox[idx + 3] * rpn_bbox_std_dev[3];

      pre_nms_anchors_[i][0] = anchors[idx + 0];
      pre_nms_anchors_[i][1] = anchors[idx + 1];
      pre_nms_anchors_[i][2] = anchors[idx + 2];
      pre_nms_anchors_[i][3] = anchors[idx + 3];
    }
    vector<vector<Dtype>> boxes =
        apply_box_deltas_graph(pre_nms_anchors_, deltas_);

    vector<Dtype> window{0, 0, 1, 1};
    vector<vector<Dtype>> boxes_ = clip_boxes_graph(boxes, window);

    vector<int> indices =
        image_nms(boxes_, nms_order, proposal_count, nms_threshold);

    //			vector<vector<Dtype>> proposals(proposal_count ,
    //vector<Dtype>(boxes_[0].size(),0.0));
    for (int i = 0; i < indices.size(); i++) {
      for (int j = 0; j < boxes_[0].size(); j++) {
        int idx =
            b * indices.size() * boxes_[0].size() + i * boxes_[0].size() + j;
        rpn_rois[idx] = boxes_[indices.at(i)][j];
      }
    }
  }
  return rpn_rois;
}
template <typename Dtype>
void MaskRCNNProposalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *rpn_class = bottom[0]->cpu_data();
  const Dtype *rpn_bbox = bottom[1]->cpu_data();
  const Dtype *anchors = bottom[2]->cpu_data();
  Dtype *rpn_rois = maskrcnnproposal(rpn_class, rpn_bbox, anchors);

  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  const int count = top[0]->count();
  caffe_copy(count, rpn_rois, top_data);
}

INSTANTIATE_CLASS(MaskRCNNProposalLayer);
REGISTER_LAYER_CLASS(MaskRCNNProposal);

} // namespace caffe
