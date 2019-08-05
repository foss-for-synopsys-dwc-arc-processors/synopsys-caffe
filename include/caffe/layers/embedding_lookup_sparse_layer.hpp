#ifndef CAFFE_EMBEDDING_LOOKUP_SPARSE_LAYER_HPP_
#define CAFFE_EMBEDDING_LOOKUP_SPARSE_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * Note: implementation of tf.embedding_lookup_sparse
 * https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup_sparse
 */

template <typename Dtype>
class EmbeddingLookupSparseLayer : public Layer<Dtype> {
public:
  explicit EmbeddingLookupSparseLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "EmbeddingLookupSparse"; }
  virtual inline int MinBottomBlobs() const { return 1; }
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

  // int n_top;
  vector<int> ids_id;
  vector<int> ids_value;
  vector<int> ids_shape;
  vector<float> w_value;
  string p_strategy;
  string combiner;
  float max_norm;
  Blob<Dtype> top_pre;
};

} // namespace caffe

#endif // CAFFE_EMBEDDING_LOOKUP_SPARSE_LAYER_HPP_
