#include <cmath>
#include <vector>

#include "caffe/layers/hard_swish_layer.hpp"

namespace caffe {

template <typename Dtype> inline Dtype hard_swish(Dtype x) {
  return x * std::min(std::max(x + 3.0, 0.), 6.0) / 6.0;
}

template <typename Dtype>
void HardSwishLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = hard_swish(bottom_data[i]);
  }
}

INSTANTIATE_CLASS(HardSwishLayer);
REGISTER_LAYER_CLASS(HardSwish);

} // namespace caffe
