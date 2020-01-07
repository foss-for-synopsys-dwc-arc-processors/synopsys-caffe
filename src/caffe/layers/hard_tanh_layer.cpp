#include <math.h>
#include <vector>

#include "caffe/layers/hard_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void HardTanhLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->count(); ++i) {
    top_data[i] = (bottom_data[i] > -1) ? bottom_data[i] : Dtype(-1);
    top_data[i] = (bottom_data[i] > 1) ? Dtype(1) : top_data[i];
  }
}

INSTANTIATE_CLASS(HardTanhLayer);
REGISTER_LAYER_CLASS(HardTanh);

} // namespace caffe
