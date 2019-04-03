#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/shape_layer.hpp"

namespace caffe {
template <typename Dtype>
void ShapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->num_axes();
  CHECK_GE(count, 0) << "Layer shape is illegal.";
  vector<int> top_shape;
  top_shape.push_back(count);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ShapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->num_axes();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom[0]->shape(i);
  }
}

INSTANTIATE_CLASS(ShapeLayer);
REGISTER_LAYER_CLASS(Shape);

}  // namespace caffe
