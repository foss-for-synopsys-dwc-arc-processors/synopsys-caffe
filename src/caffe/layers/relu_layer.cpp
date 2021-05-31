#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void QuantizeLeakyRelu(const int n, const Dtype *in, Dtype *out, Dtype alpha, double in_s, int in_zp, double out_s, int out_zp) {
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/leaky_relu.h#L49-L59
  int shift_ident, shift_alpha;
  int mul_ident = tfl_QuantizeMultiplier((double)in_s / out_s, &shift_ident); // for positive value
  int mul_alpha = tfl_QuantizeMultiplier((double)in_s * (double)alpha / out_s, &shift_alpha); // for negative value
  int input_value, unclamped_output;
  for (int i = 0; i < n; ++i) {
    input_value = (int) std::round(in[i]) - in_zp;
    if (input_value >= 0) {
      unclamped_output = out_zp + tfl_MultiplyByQuantizedMultiplier(
                                    input_value, mul_ident, shift_ident);
    } else {
      unclamped_output = out_zp + tfl_MultiplyByQuantizedMultiplier(
                                    input_value, mul_alpha, shift_alpha);
    }
    // if (unclamped_output < clip_min) unclamped_output = clip_min; // will do caffe_cpu_saturate later
    // if (unclamped_output > clip_max) unclamped_output = clip_max;
    out[i] = Dtype(unclamped_output);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  Dtype relu6 = this->layer_param_.relu_param().relu6(); //CUSTOMIZATION
  Dtype maximum = this->layer_param_.relu_param().maximum(); //CUSTOMIZATION
  Dtype minimum = this->layer_param_.relu_param().minimum(); //CUSTOMIZATION
  double input_scale_ = this->layer_param_.relu_param().input_scale(); //CUSTOMIZATION
  double output_scale_ = this->layer_param_.relu_param().output_scale(); //CUSTOMIZATION
  int input_zero_point_ = this->layer_param_.relu_param().input_zero_point(); //CUSTOMIZATION
  int output_zero_point_ = this->layer_param_.relu_param().output_zero_point(); //CUSTOMIZATION
  Dtype saturate_ = this->layer_param_.relu_param().saturate(); //CUSTOMIZATION
  if (bottom.size() > 1)  //bottom[1] provides the maximum case
  	maximum = bottom[1]->cpu_data()[0];
  const bool quant_in = (input_scale_ != Dtype(1.0) || input_zero_point_ != 0);
  const bool quant_out = (output_scale_ != Dtype(1.0) || output_zero_point_ != 0);
  if (negative_slope != Dtype(0) && quant_in && quant_out) {
    QuantizeLeakyRelu(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data(),
      negative_slope, input_scale_, input_zero_point_, output_scale_, output_zero_point_);
    caffe_cpu_saturate(top[0]->count(), top[0]->mutable_cpu_data(), saturate_); // if None nothing happens
    return;
  }
  if (quant_in) {
      caffe_cpu_dequantize<Dtype>(bottom[0]->count(), bottom[0]->mutable_cpu_data(),
          input_scale_, input_zero_point_);
  }
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
    if(isnan(top_data[i])) //CUSTOMIZATION
      top_data[i] = 0;
    if(relu6) //CUSTOMIZATION
      top_data[i] = std::min(top_data[i], Dtype(6)); //CUSTOMIZATION
    if(maximum > Dtype(0))
      top_data[i] = std::min(top_data[i], maximum); //CUSTOMIZATION
    if(minimum != Dtype(0))
      top_data[i] = std::max(top_data[i], minimum); //CUSTOMIZATION
  }

  if (quant_out) {
    // do not reuse "top_data"; it is shifted during the computation
    caffe_cpu_quantize<Dtype>(top[0]->count(), top[0]->mutable_cpu_data(), output_scale_, output_zero_point_);
  }
  if (quant_in) {
    caffe_cpu_quantize<Dtype>(bottom[0]->count(), bottom[0]->mutable_cpu_data(),
        input_scale_, input_zero_point_);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    Dtype relu6 = this->layer_param_.relu_param().relu6(); //CUSTOMIZATION
    Dtype maximum = this->layer_param_.relu_param().maximum(); //CUSTOMIZATION
    Dtype minimum = this->layer_param_.relu_param().minimum(); //CUSTOMIZATION
    if (bottom.size() > 1)  //bottom[1] provides the maximum case
      maximum = bottom[1]->cpu_data()[0];
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
      if(relu6) //CUSTOMIZATION
        bottom_diff[i] *= (bottom_data[i] < Dtype(6));
      if(maximum > Dtype(0)) //CUSTOMIZATION
        bottom_diff[i] *= (bottom_data[i] < maximum);
      if(minimum != Dtype(0)) //CUSTOMIZATION
        bottom_diff[i] *= (bottom_data[i] > minimum);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
