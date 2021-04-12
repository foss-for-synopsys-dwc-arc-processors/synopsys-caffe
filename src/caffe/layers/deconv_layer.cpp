#include <vector>

#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
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
    //<--CUSTOMIZATION
    int output_dim;
    if (pad_l!=0 || pad_r!=0 || pad_t!=0 || pad_b!=0){ //only support 2D
      if (i==0) {
        output_dim = stride_data[i] * (input_dim - 1)
            + kernel_extent - pad_t - pad_b;
      }
      if (i==1) {
        output_dim = stride_data[i] * (input_dim - 1)
            + kernel_extent - pad_l - pad_r;
      }
    }
    else{
      output_dim = stride_data[i] * (input_dim - 1)
          + kernel_extent - 2 * pad_data[i];
    }
    //CUSTOMIZATION-->
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
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
      caffe_cpu_dequantize<Dtype>(bottom[i]->count(), bottom[i]->mutable_cpu_data(), input_scale, input_zero_point);
    }
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
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
    if (quant_in) {
      caffe_cpu_quantize<Dtype>(bottom[i]->count(), bottom[i]->mutable_cpu_data(), input_scale, input_zero_point);
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
void DeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // Gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(top_diff + n * this->top_dim_,
              bottom_data + n * this->bottom_dim_, weight_diff);
        }
        // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
        // we might have just computed above.
        if (propagate_down[i]) {
          this->forward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_,
              this->param_propagate_down_[0]);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DeconvolutionLayer);
#endif

INSTANTIATE_CLASS(DeconvolutionLayer);

}  // namespace caffe
