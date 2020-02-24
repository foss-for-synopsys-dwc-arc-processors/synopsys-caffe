#ifndef CAFFE_POOLING3D_LAYER_HPP_
#define CAFFE_POOLING3D_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype> class Pooling3DLayer : public Layer<Dtype> {
public:
  explicit Pooling3DLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Pooling3D"; }
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

  int kernel_h_, kernel_w_, kernel_d_;
  int stride_h_, stride_w_, stride_d_;
  int pad_h0_, pad_w0_, pad_d0_, pad_h1_, pad_w1_, pad_d1_;
  int num_, channels_;
  int height_, width_, depth_;
  int pooled_height_, pooled_width_, pooled_depth_;
  bool global_pooling_;
  bool ceil_mode_;

};

} // namespace caffe

#endif // CAFFE_POOLING3D_LAYER_HPP_
