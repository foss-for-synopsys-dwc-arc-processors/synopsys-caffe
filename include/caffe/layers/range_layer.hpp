#ifndef CAFFE_RANGE_LAYER_HPP_
#define CAFFE_RANGE_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * Note: implementation of tf.range
 * https://www.tensorflow.org/api_docs/python/tf/range
 */

template <typename Dtype> class RangeLayer : public Layer<Dtype> {
public:
  explicit RangeLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Range"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }

  float start_;
  float limit_;
  float delta_;
  vector<float> range_;
};

} // namespace caffe

#endif // CAFFE_RANGE_LAYER_HPP_
