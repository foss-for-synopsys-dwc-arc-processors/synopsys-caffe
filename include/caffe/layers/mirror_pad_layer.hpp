#ifndef CAFFE_MIRROR_PAD_LAYER_HPP_
#define CAFFE_MIRROR_PAD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MirrorPadLayer : public Layer<Dtype> {
 public:

  explicit MirrorPadLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MirrorPad"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  inline vector<int> indices(int offset, const vector<int> & shape) const;
  inline int offset(const vector<int>& indices, const vector<int> & shape) const;

  vector<int> paddings_;
  float constant_values_;
  string mode_;
};

}  // namespace caffe

#endif  // CAFFE_MIRROR_PAD_LAYER_HPP_
