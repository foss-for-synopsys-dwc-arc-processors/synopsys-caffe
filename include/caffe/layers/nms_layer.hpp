#ifndef CAFFE_NMS_LAYER_HPP_
#define CAFFE_NMS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief TensorFlow style implementation of NMS.
 *
 * ref: https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
 */

template <typename Dtype>
class NMSLayer : public Layer<Dtype> {
 public:

  explicit NMSLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NMS"; }
  //virtual inline int MinBottomBlobs() const { return 1; }
  //virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  virtual void apply_nms(vector<vector<Dtype> > &pred_boxes, vector<int> &indices, float iou_threshold);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  size_t top_k_;
  bool has_axis_;
  int axis_;
  float iou_threshold_;
};

}  // namespace caffe

#endif  // CAFFE_NMS_LAYER_HPP_
