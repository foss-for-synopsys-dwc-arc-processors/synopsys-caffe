#include <math.h>
#include <vector>

#include "caffe/layers/softplus_layer.hpp"

namespace caffe {


template <typename Dtype>
void SoftplusLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = logf(expf(bottom_data[i]) + 1);
  }
}

INSTANTIATE_CLASS(SoftplusLayer);
REGISTER_LAYER_CLASS(Softplus);

}  // namespace caffe
