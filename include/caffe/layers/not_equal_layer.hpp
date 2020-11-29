#ifndef CAFFE_NOT_EQUAL_LAYER_HPP_
#define CAFFE_NOT_EQUAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	// implement of Tensorflow Operator: https://www.tensorflow.org/api_docs/python/tf/math/not_equal


template <typename Dtype> class NotEqualLayer : public Layer<Dtype> {
public:
  explicit NotEqualLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
	  const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char *type() const { return "NotEqual"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	  NOT_IMPLEMENTED;
  };

  float comparand_;
  int const_flag_;
};

} // namespace caffe

#endif // CAFFE_NOT_EQUAL_LAYER_HPP_

