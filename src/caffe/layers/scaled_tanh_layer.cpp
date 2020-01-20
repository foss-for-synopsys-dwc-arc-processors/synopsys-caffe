#include <math.h>
#include <vector>

#include "caffe/layers/scaled_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScaledTanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const float alpha = this->layer_param_.scaled_tanh_param().alpha();
  const float beta = this->layer_param_.scaled_tanh_param().beta();
  for (int i = 0; i < bottom[0]->count(); ++i) {
    top_data[i] = alpha * tanh(beta * bottom_data[i]);
  }
}

INSTANTIATE_CLASS(ScaledTanHLayer);
REGISTER_LAYER_CLASS(ScaledTanH);

} // namespace caffe
