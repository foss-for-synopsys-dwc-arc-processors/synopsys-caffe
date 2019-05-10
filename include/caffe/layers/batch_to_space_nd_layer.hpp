#ifndef CAFFE_BATCHTOSPACEND_LAYER_HPP_
#define CAFFE_BATCHTOSPACEND_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BatchToSpaceNDLayer : public Layer<Dtype> {
 public:

  explicit BatchToSpaceNDLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BatchToSpaceND"; }
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
private:
  inline vector<int> indices(int offset, const vector<int> & shape) const;
  inline int offset(const vector<int>& indices, const vector<int> & shape) const;

  vector<int> block_shape_;
  vector<int> crops_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHTOSPACEND_LAYER_HPP_
