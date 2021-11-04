#ifndef CAFFE_SPATIAL_BATCHING_POOLING_LAYER_HPP_
#define CAFFE_SPATIAL_BATCHING_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SpatialBatchingPoolingLayer : public Layer<Dtype> {
public:
  explicit SpatialBatchingPoolingLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "SpatialBatchingPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.spatial_batching_pooling_param().pool() ==
            SpatialBatchingPoolingParameter_PoolMethod_MAX)
               ? 2
               : 1;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
  bool ceil_mode_;
  int pad_l_;      // CUSTOMIZATION
  int pad_r_;      // CUSTOMIZATION
  int pad_t_;      // CUSTOMIZATION
  int pad_b_;      // CUSTOMIZATION
  Dtype saturate_; // CUSTOMIZATION
  double input_scale_; //CUSTOMIZATION
  int input_zero_point_; //CUSTOMIZATION
  double output_scale_; //CUSTOMIZATION
  int output_zero_point_; //CUSTOMIZATION

  int spatial_batching_h_;
  int spatial_batching_w_;
  int skip_h_, skip_w_;
  int batch_h_, batch_w_;
  int gap_h_, gap_w_;
  int pooled_batch_h_;
  int pooled_batch_w_;
  int pooled_gap_h_;
  int pooled_gap_w_;
  int s_pooled_height_, s_pooled_width_;
};

} // namespace caffe

#endif // CAFFE_SPATIAL_BATCHING_POOLING_LAYER_HPP_
