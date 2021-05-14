#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
  input_scale_ = this->layer_param_.softmax_param().input_scale(); //CUSTOMIZATION
  output_scale_ = this->layer_param_.softmax_param().output_scale(); //CUSTOMIZATION
  input_zero_point_ = this->layer_param_.softmax_param().input_zero_point(); //CUSTOMIZATION
  output_zero_point_ = this->layer_param_.softmax_param().output_zero_point(); //CUSTOMIZATION
  saturate_ = this->layer_param_.softmax_param().saturate(); //CUSTOMIZATION
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const bool quant_in = (input_scale_ != Dtype(1.0) || input_zero_point_ != 0);
  const bool quant_out = (output_scale_ != Dtype(1.0) || output_zero_point_ != 0);
  /* For quantized softmax, tflite computes with float numbers. Refer to 
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/optimized/optimized_ops.h#L3765
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/softmax_quantized_test.cc#L49-L53
  */
  if (quant_in) {
    caffe_cpu_dequantize<Dtype>(bottom[0]->count(), bottom[0]->mutable_cpu_data(),
        input_scale_, input_zero_point_);
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
  if (quant_out) {
    const int count_t = top[0]->count();
    top_data = top[0]->mutable_cpu_data();
    // reset "top_data"; it is shifted during the computation
    caffe_cpu_quantize<Dtype>(count_t, top_data, output_scale_, output_zero_point_);
    // uint8_256 represents float_1, and saturate clamps it to 255.
    if (saturate_ == SoftmaxParameter_SaturateMethod_Signed)
      caffe_cpu_signed_saturate(count_t, top_data);
    if (saturate_ == SoftmaxParameter_SaturateMethod_Unsigned)
      caffe_cpu_unsigned_saturate(count_t, top_data);
    if (saturate_ == SoftmaxParameter_SaturateMethod_Signed_8bit)
      caffe_cpu_signed_8bit_saturate(count_t, top_data);
    if (saturate_ == SoftmaxParameter_SaturateMethod_Unsigned_8bit)
      caffe_cpu_unsigned_8bit_saturate(count_t, top_data);
  }
  if (quant_in) {
    caffe_cpu_quantize<Dtype>(bottom[0]->count(), bottom[0]->mutable_cpu_data(),
        input_scale_, input_zero_point_);
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
