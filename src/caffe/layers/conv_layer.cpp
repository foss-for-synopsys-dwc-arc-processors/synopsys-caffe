#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int pad_type = this->pad_type_; //CUSTOMIZATION
  const int pad_l = this->pad_l_; //CUSTOMIZATION
  const int pad_r = this->pad_r_; //CUSTOMIZATION
  const int pad_t = this->pad_t_; //CUSTOMIZATION
  const int pad_b = this->pad_b_; //CUSTOMIZATION
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    int output_dim;
    //<--CUSTOMIZATION
    if (pad_l!=0 || pad_r!=0 || pad_t!=0 || pad_b!=0){ //only support 2D
      if (i==0) {
        output_dim = (input_dim + pad_t + pad_b - kernel_extent) / stride_data[i] + 1;
      }
      if (i==1) {
        output_dim = (input_dim + pad_l + pad_r - kernel_extent) / stride_data[i] + 1;
      }
    }
    else{
      switch (pad_type) {
        case 0:
          output_dim = (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
          break;
        case 1:
          output_dim = ceil(float(input_dim) / float(stride_data[i]));
          break;
        default:
          LOG(FATAL)<< "Unknown padding type.";
          break;
      }
    //CUSTOMIZATION-->
    }
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // set up quantization parameters: scale + zero_point
  const Dtype input_scale = this->input_scale_;
  const Dtype output_scale = this->output_scale_;
  const Dtype weight_scale = this->weight_scale_;
  const Dtype bias_scale = this->bias_scale_;
  const int input_zero_point = this->input_zero_point_;
  const int output_zero_point = this->output_zero_point_;
  const int weight_zero_point = this->weight_zero_point_;
  const int bias_zero_point = this->bias_zero_point_;
  const Dtype saturate = this->saturate_;
  const bool quant_in = (input_scale != Dtype(1.0) || input_zero_point != 0);
  const bool quant_out = (output_scale != Dtype(1.0) || output_zero_point != 0);
  const bool quant_w = (weight_scale != Dtype(1.0) || weight_zero_point != 0);
  const bool quant_b = (this->bias_term_&& (bias_scale != Dtype(1.0) || bias_zero_point != 0));
  if (quant_w) {
    Dtype *qw = this->blobs_[0]->mutable_cpu_data();
    caffe_cpu_dequantize<Dtype>(this->blobs_[0]->count(), qw, weight_scale, weight_zero_point);
  }
  if (quant_b) {
    Dtype *qb = this->blobs_[1]->mutable_cpu_data();
    caffe_cpu_dequantize<Dtype>(this->blobs_[1]->count(), qb, bias_scale, bias_zero_point);
  }

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    if (quant_in) {
      Dtype* qin = bottom[i]->mutable_cpu_data();
      caffe_cpu_dequantize<Dtype>(bottom[i]->count(), qin, input_scale, input_zero_point);
    }

    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }

    const int count_t = top[i]->count();
    if (quant_out) {
      caffe_cpu_quantize<Dtype>(count_t, top_data, output_scale, output_zero_point);
    }
    if (saturate == ConvolutionParameter_SaturateMethod_Signed)
      caffe_cpu_signed_saturate(count_t, top_data);
    if (saturate == ConvolutionParameter_SaturateMethod_Unsigned)
      caffe_cpu_unsigned_saturate(count_t, top_data);
    if (saturate == ConvolutionParameter_SaturateMethod_Signed_8bit)
      caffe_cpu_signed_8bit_saturate(count_t, top_data);
    if (saturate == ConvolutionParameter_SaturateMethod_Unsigned_8bit)
      caffe_cpu_unsigned_8bit_saturate(count_t, top_data);

    if (quant_in) { // restore the quantized input blob
      Dtype* qin = bottom[i]->mutable_cpu_data();
      caffe_cpu_quantize<Dtype>(bottom[i]->count(), qin, input_scale, input_zero_point);
    }
  }
  // restore quantized weight/bias
  if (quant_w) {
    Dtype *qw = this->blobs_[0]->mutable_cpu_data();
    caffe_cpu_quantize<Dtype>(this->blobs_[0]->count(), qw, weight_scale, weight_zero_point);
  }
  if (quant_b) {
    Dtype *qb = this->blobs_[1]->mutable_cpu_data();
    caffe_cpu_quantize<Dtype>(this->blobs_[1]->count(), qb, bias_scale, bias_zero_point);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  //default update_weight =true
  bool update_weight = !this->layer_param_.convolution_param().weight_fixed();
  if (this->layer_param_.convolution_param().gen_mode() && this->gan_mode_ != 2) {
	update_weight = false;
  }
  if (this->layer_param_.convolution_param().dis_mode() && this->gan_mode_ == 2) {
	update_weight = false;
  }
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1] && update_weight) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0] && update_weight) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  // update gan_mode_
  this->gan_mode_ = this->gan_mode_ == 2 ? 1 : this->gan_mode_ + 1;
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
