#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/range_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RangeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  const RangeParameter &range_param = this->layer_param_.range_param();
  start_ = range_param.start();
  delta_ = range_param.delta();
  limit_ = range_param.limit();
  CHECK_NE(delta_, 0) << "requires delta != 0";
  float num = start_;
  if (delta_ > 0) {
    while (num < limit_) {
      range_.insert(range_.end(), num);
      num += delta_;
    }
  } else {
    while (num > limit_) {
      range_.insert(range_.end(), num);
      num += delta_;
    }
  }
}

template <typename Dtype>
void RangeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {
  int length = range_.size();
  vector<int> len = {length};
  top[0]->Reshape(len);
}

template <typename Dtype>
void RangeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  Dtype *top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < range_.size(); ++i) {
    top_data[i] = range_[i];
  }
}

INSTANTIATE_CLASS(RangeLayer);
REGISTER_LAYER_CLASS(Range);

} // namespace caffe
