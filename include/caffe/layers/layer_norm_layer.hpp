#ifndef CAFFE_LAYERNORM_LAYER_HPP_
#define CAFFE_LAYERNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 */
template <typename Dtype> class LayerNormLayer : public Layer<Dtype> {
public:
  explicit LayerNormLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "LayerNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }

  Blob<Dtype> mean_, variance_, temp_;
  Blob<Dtype> flat_sum_multiplier_;
  Blob<Dtype> batch_sum_multiplier_;
  Dtype eps_;
  bool elementwise_affine_;
};

} // namespace caffe

#endif // CAFFE_LAYERNORM_LAYER_HPP_
