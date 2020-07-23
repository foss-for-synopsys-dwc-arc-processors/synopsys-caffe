#ifndef CAFFE_NON_MAX_SUPPRESSION_LAYER_HPP_
#define CAFFE_NON_MAX_SUPPRESSION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief ONNX style implementation of NonMaxSuppression.
 *
 * Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
 * Bounding boxes with score less than score_threshold are removed.
 * Bounding box format is indicated by attribute center_point_box.
 * ref: https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression
 */

template <typename Dtype>
class NonMaxSuppressionLayer : public Layer<Dtype> {
 public:

  explicit NonMaxSuppressionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NonMaxSuppression"; }
  //virtual inline int MinBottomBlobs() const { return 1; }
  //virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  virtual bool SuppressByIOU(const Dtype* boxes_data, int64_t box_index1,
      int64_t box_index2, int64_t center_point_box, float iou_threshold);
  virtual void MaxMin(float lhs, float rhs, float& min, float& max);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }


  struct ScoreIndexPair {
    Dtype score_{};
    int64_t index_{};

    ScoreIndexPair() = default;
    explicit ScoreIndexPair(Dtype score, int64_t idx) : score_(score), index_(idx) {}

    bool operator<(const ScoreIndexPair& rhs) const {
      return score_ < rhs.score_;
    }
  };

  int max_output_boxes_per_class_;
  float iou_threshold_;
  float score_threshold_;
  int center_point_box_;

  int num_batches_;
  int num_classes_;
  int num_boxes_ ;
};

}  // namespace caffe

#endif  // CAFFE_NON_MAX_SUPPRESSION_LAYER_HPP_
