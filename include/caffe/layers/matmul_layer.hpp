#ifndef CAFFE_MATMUL_LAYER_HPP_
#define CAFFE_MATMUL_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * @brief Resize images to size using nearest neighbor interpolation. ////
 * Note: implementation of tf.linalg.matmul()
 * https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/linalg/matmul
 */

template <typename Dtype> class MatMulLayer : public Layer<Dtype> {
public:
  explicit MatMulLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "MatMul"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
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
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top) {}
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  //    {}

  int num_axes;
  int M;
  int N;
  int K;
  bool transpose_a;
  bool transpose_b;
};

} // namespace caffe

#endif // CAFFE_MATMUL_LAYER_HPP_
