#ifndef CAFFE_SPARSE_TO_DENSE_LAYER_HPP_
#define CAFFE_SPARSE_TO_DENSE_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * Note: implementation of tf.sparse_to_dense()
 * https://www.tensorflow.org/api_docs/python/tf/sparse_to_dense
 */

template <typename Dtype> class SparseToDenseLayer : public Layer<Dtype> {
public:
  explicit SparseToDenseLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "SparseToDense"; }
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

  inline int count(const vector<int> &shape, const int axis) const;
  vector<int> sparse_indices_;
  vector<int> sparse_indices_shape_;
  vector<int> output_shape_;
  vector<float> sparse_values_;
  float default_value_;
  bool validate_indices_;
};

} // namespace caffe

#endif // CAFFE_SPARSE_TO_DENSE_LAYER_HPP_
