#include <math.h>
#include <vector>

#include "caffe/layers/thresholded_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ThresholdedReluLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const float alpha = this->layer_param_.thresholded_relu_param().alpha();
  for (int i = 0; i < bottom[0]->count(); ++i) {
    top_data[i] = (bottom_data[i] > alpha) ? bottom_data[i] : Dtype(0);
  }
}

INSTANTIATE_CLASS(ThresholdedReluLayer);
REGISTER_LAYER_CLASS(ThresholdedRelu);

} // namespace caffe
