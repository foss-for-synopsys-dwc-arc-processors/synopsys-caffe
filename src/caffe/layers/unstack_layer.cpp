#include <algorithm>
#include <vector>

#include "caffe/layers/unstack_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void UnstackLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const UnstackParameter &unstack_param = this->layer_param_.unstack_param();
  unstack_axis_ = bottom[0]->CanonicalAxisIndex(unstack_param.axis());
  unstack_num_ = unstack_param.num();
  if (unstack_num_ == 0)
    unstack_num_ = bottom[0]->shape(unstack_axis_);
}

template <typename Dtype>
void UnstackLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  // const UnstackParameter& unstack_param = this->layer_param_.unstack_param();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.erase(top_shape.begin() + unstack_axis_);
  for (int i = 0; i < top.size(); ++i)
    top[i]->Reshape(top_shape);
}

template <typename Dtype>
void UnstackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const int size_unstack_axis = bottom[0]->shape(unstack_axis_);
  vector<int> bottom_shape = bottom[0]->shape();
  int strides = 1;
  for (int i = unstack_axis_ + 1; i < bottom_shape.size(); i++)
    strides *= bottom_shape[i];
  int num_unstack_ = bottom[0]->count() / strides / size_unstack_axis;
  // num_unstack_ /= size_unstack_axis;
  for (int i = 0; i < top.size(); ++i) {
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < num_unstack_; ++n) {
      const int top_offset = n * strides;
      const int bottom_offset = (n * size_unstack_axis + i) * strides;
      caffe_copy(strides, bottom_data + bottom_offset, top_data + top_offset);
    }
  }
}

INSTANTIATE_CLASS(UnstackLayer);
REGISTER_LAYER_CLASS(Unstack);

} // namespace caffe
