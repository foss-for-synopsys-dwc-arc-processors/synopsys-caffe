#include <vector>

#include "caffe/layers/deconv_layer.hpp"
#define W this->blobs_[0]
#define B this->blobs_[1]

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
  const double input_scale = this->input_scale_;
  const double output_scale = this->output_scale_;
  const double weight_scale = this->weight_scale_;
  const int input_zero_point = this->input_zero_point_;
  const int output_zero_point = this->output_zero_point_;
  const int weight_zero_point = this->weight_zero_point_;
  const Dtype saturate = this->saturate_;
  const bool shift_input = (input_zero_point != 0);
  const bool shift_weight = (weight_zero_point != 0);
  const bool scale_output = (input_scale != Dtype(1.0) || weight_scale != Dtype(1.0) ||
                             output_scale != Dtype(1.0));
  const bool shift_output = (output_zero_point != 0);
  
  if (shift_weight) { // shift the quantized weight
    caffe_add_scalar<Dtype>(W->count(), Dtype(-weight_zero_point), W->mutable_cpu_data());
  }

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    if (shift_input) {
      caffe_add_scalar<Dtype>(bottom[i]->count(),
        Dtype(-input_zero_point), bottom[i]->mutable_cpu_data());
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
    if (scale_output) {
      Dtype out_scal = double(input_scale * weight_scale) / output_scale;
      caffe_cpu_scale_double_round<Dtype>(count_t, out_scal, top_data);
    }
    if (shift_output) {
      caffe_add_scalar<Dtype>(count_t, Dtype(output_zero_point), top_data);
    }
    if (saturate == ConvolutionParameter_SaturateMethod_Signed)
      caffe_cpu_signed_saturate(count_t, top_data);
    if (saturate == ConvolutionParameter_SaturateMethod_Unsigned)
      caffe_cpu_unsigned_saturate(count_t, top_data);
    if (saturate == ConvolutionParameter_SaturateMethod_Signed_8bit)
      caffe_cpu_signed_8bit_saturate(count_t, top_data);
    if (saturate == ConvolutionParameter_SaturateMethod_Unsigned_8bit)
      caffe_cpu_unsigned_8bit_saturate(count_t, top_data);
    if (shift_input) { // shift the quantized input blob back to correct range
      caffe_add_scalar<Dtype>(bottom[i]->count(),
        Dtype(input_zero_point), bottom[i]->mutable_cpu_data());
    }
  }
  // shift quantized weight/bias back to correct range
  if (shift_weight) {
    caffe_add_scalar<Dtype>(W->count(), Dtype(weight_zero_point), W->mutable_cpu_data());
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
