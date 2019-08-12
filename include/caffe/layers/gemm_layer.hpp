#ifndef CAFFE_GEMM_LAYER_HPP_
#define CAFFE_GEMM_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * @brief Resize images to size using nearest neighbor interpolation. ////
 * Note: implementation of onnx gemm
 * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
 */

template <typename Dtype> class GemmLayer : public Layer<Dtype> {
public:
  explicit GemmLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Gemm"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
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
  Dtype alpha_;
  Dtype beta_;
  int transa;
  int transb;
  int M_;
  int N_;
  int K_;
};

} // namespace caffe

#endif // CAFFE_GEMM_LAYER_HPP_
