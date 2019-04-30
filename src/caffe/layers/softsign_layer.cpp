#include <math.h>
#include <vector>

#include "caffe/layers/softsign_layer.hpp"

namespace caffe {


template <typename Dtype>
void SoftsignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] /(1.0 + fabs(bottom_data[i]));
  }
}

INSTANTIATE_CLASS(SoftsignLayer);
REGISTER_LAYER_CLASS(Softsign);

}  // namespace caffe
