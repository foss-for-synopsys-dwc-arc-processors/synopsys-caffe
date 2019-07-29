#ifndef CAFFE_EMBEDDING_LOOKUP_LAYER_HPP_
#define CAFFE_EMBEDDING_LOOKUP_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * Note: implementation of tf.embedding_lookup
 * https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
 */

template <typename Dtype> class EmbeddingLookupLayer : public Layer<Dtype> {
public:
  explicit EmbeddingLookupLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "EmbeddingLookup"; }
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
  vector<int> ids;
  vector<int> ids_shape;
  string p_strategy;
  float max_norm;
};

} // namespace caffe

#endif // CAFFE_EMBEDDING_LOOKUP_LAYER_HPP_
