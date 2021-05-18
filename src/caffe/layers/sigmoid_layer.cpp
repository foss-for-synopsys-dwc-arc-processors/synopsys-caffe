#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SigmoidParameter& sigmoid_param = this->layer_param_.sigmoid_param();
  input_scale_ = sigmoid_param.input_scale();
  output_scale_ = sigmoid_param.output_scale();
  input_zero_point_ = sigmoid_param.input_zero_point();
  output_zero_point_ = sigmoid_param.output_zero_point();
  saturate_ = sigmoid_param.saturate();
}

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const bool quant_in = input_scale_ != Dtype(1.0) || input_zero_point_ != 0;
  const bool quant_out = output_scale_ != Dtype(1.0) || output_zero_point_ != 0;
  if (quant_in) {
    caffe_cpu_dequantize<Dtype>(bottom[0]->count(), bottom[0]->mutable_cpu_data(),
        input_scale_, input_zero_point_);
  } // CUSTOMIZATION
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
  if (quant_out) {
    caffe_cpu_quantize<Dtype>(count, top_data,
        output_scale_, output_zero_point_);
    if (saturate_ == SigmoidParameter_SaturateMethod_Signed)
      caffe_cpu_signed_saturate(count, top_data);
    if (saturate_ == SigmoidParameter_SaturateMethod_Unsigned)
      caffe_cpu_unsigned_saturate(count, top_data);
    if (saturate_ == SigmoidParameter_SaturateMethod_Signed_8bit)
      caffe_cpu_signed_8bit_saturate(count, top_data);
    if (saturate_ == SigmoidParameter_SaturateMethod_Unsigned_8bit)
      caffe_cpu_unsigned_8bit_saturate(count, top_data);
  } // CUSTOMIZATION
  if (quant_in) {
    caffe_cpu_quantize<Dtype>(bottom[0]->count(), bottom[0]->mutable_cpu_data(),
        input_scale_, input_zero_point_);
  } // CUSTOMIZATION
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
