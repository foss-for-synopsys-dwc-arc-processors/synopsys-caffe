#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

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
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  Dtype relu6 = this->layer_param_.relu_param().relu6(); //CUSTOMIZATION
  Dtype maximum = this->layer_param_.relu_param().maximum(); //CUSTOMIZATION
  Dtype minimum = this->layer_param_.relu_param().minimum(); //CUSTOMIZATION
  Dtype input_scale_ = this->layer_param_.relu_param().input_scale(); //CUSTOMIZATION
  Dtype output_scale_ = this->layer_param_.relu_param().output_scale(); //CUSTOMIZATION
  int input_zero_point_ = this->layer_param_.relu_param().input_zero_point(); //CUSTOMIZATION
  int output_zero_point_ = this->layer_param_.relu_param().output_zero_point(); //CUSTOMIZATION
  if (bottom.size() > 1)  //bottom[1] provides the maximum case
  	maximum = bottom[1]->cpu_data()[0];
  const bool quant_in = (input_scale_ != Dtype(1.0) || input_zero_point_ != 0);
  const bool quant_out = (output_scale_ != Dtype(1.0) || output_zero_point_ != 0);
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
