#include <math.h>
#include <vector>

#include "caffe/layers/hard_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void HardSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const float alpha = this->layer_param_.hard_sigmoid_param().alpha();
  const float beta = this->layer_param_.hard_sigmoid_param().beta();
  for (int i = 0; i < bottom[0]->count(); ++i) {
    Dtype result = alpha * bottom_data[i] + beta;
    if (result <= 0) {
      top_data[i] = 0;
    } else {
      top_data[i] = (result > 1) ? Dtype(1) : result;
    }
  }
}

INSTANTIATE_CLASS(HardSigmoidLayer);
REGISTER_LAYER_CLASS(HardSigmoid);

} // namespace caffe
