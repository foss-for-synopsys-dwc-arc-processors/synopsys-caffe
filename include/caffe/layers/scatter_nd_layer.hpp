#ifndef CAFFE_SCATTER_ND_LAYER_HPP_
#define CAFFE_SCATTER_ND_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class ScatterNDLayer : public Layer<Dtype> {
 public:
  explicit ScatterNDLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScatterND"; }
  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  bool is_data_param_;
  bool is_indices_param_;
  bool is_updates_param_;
  int Q_; // rank of indices
  int K_; // each entry of indices tensor is a K-tuple = partial-index of data tensor
};

}  // namespace caffe

#endif  // CAFFE_SCATTER_ND_LAYER_HPP_
