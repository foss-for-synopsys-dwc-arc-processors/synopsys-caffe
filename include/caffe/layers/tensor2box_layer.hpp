#ifndef CAFFE_TENSOR2BOX_LAYER_HPP_
#define CAFFE_TENSOR2BOX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Tensor2BoxLayer : public Layer<Dtype> {
public:
  explicit Tensor2BoxLayer(const LayerParameter& param) :
      Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {return "Tensor2Box";}
  virtual inline int ExactBottomBlobs() const {return 1;}
  virtual inline int ExactNumTopBlobs() const {return 1;}

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int num_classes_;
  int img_dim_h_, img_dim_w_;
  vector<int> anchors_x_, anchors_y_;
  // We have a python example such that
  // anchor == [(10, 10), (10, 20), (50, 50)]
  // num_classes == 4
  // img_dim == (320, 480)
};

}  // namespace caffe

#endif  // CAFFE_TENSOR2BOX_LAYER_HPP_
