#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype input_scale = this->input_scale_; //CUSTOMIZATION
  Dtype output_scale = this->output_scale_; //CUSTOMIZATION
  int input_zero_point = this->input_zero_point_; //CUSTOMIZATION
  int output_zero_point = this->output_zero_point_; //CUSTOMIZATION
  Dtype saturate = this->saturate_; //CUSTOMIZATION
  for (int i = 0; i < bottom.size(); ++i) {
    Dtype* bottom_data = bottom[i]->mutable_gpu_data();
    //<--CUSTOMIZATION
    const int count_b = bottom[i]->count();
    /*** Denote input_scale=s0,input_zero_point=z0,input_blob=x0;
                output_scale=s1,output_zero_point=z1;
                Weight=W0, Bias=B0, X=Convolution
        ( (x0-z0)*s0 X W0 + B0 ) / s1 + z1
      = ( (x0-z0) X W0 + B0/S0)) * s0/s1 + z1
      Tried both computation, neither achieve bit-wise precision referring to Caffe2
    ***/
    if (input_zero_point != 0) {
      caffe_gpu_add_scalar(count_b, Dtype(-input_zero_point), bottom_data);
    }
    if (input_scale != Dtype(1)) {
      caffe_gpu_scal(count_b, input_scale, bottom_data);
    }
    //CUSTOMIZATION-->
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    //<--CUSTOMIZATION
    const int count_t = top[i]->count();
    if (output_scale != Dtype(1)) {
      caffe_gpu_scal(count_t, output_scale, top_data);
      caffe_gpu_round(count_t, top_data);
    }
    if (output_zero_point != 0) {
      caffe_gpu_add_scalar(count_t, Dtype(output_zero_point), top_data);
    }
    if(saturate ==  ConvolutionParameter_SaturateMethod_Signed)
      caffe_gpu_signed_saturate(count_t, top_data);
    if(saturate ==  ConvolutionParameter_SaturateMethod_Unsigned)
      caffe_gpu_unsigned_saturate(count_t, top_data);
    if(saturate ==  ConvolutionParameter_SaturateMethod_Signed_8bit)
      caffe_gpu_signed_8bit_saturate(count_t, top_data);
    if(saturate ==  ConvolutionParameter_SaturateMethod_Unsigned_8bit)
      caffe_gpu_unsigned_8bit_saturate(count_t, top_data);

    // retrieve the quantized bottom blobs
    // in case some other layer consumes the same input blob
    if (input_scale != Dtype(1)) {
      caffe_gpu_scal(count_b, Dtype(1.0) / input_scale, bottom_data);
      caffe_gpu_round(count_b, bottom_data);
    }
    if (input_zero_point != 0) {
      caffe_gpu_add_scalar(count_b, Dtype(input_zero_point), bottom_data);
    }
    //caffe_gpu_unsigned_8bit_saturate(count_b, bottom_data);
    //CUSTOMIZATION-->
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  bool update_weight = !this->layer_param_.convolution_param().weight_fixed();
  if (this->layer_param_.convolution_param().gen_mode() && this->gan_mode_ != 2 ) {
	update_weight = false;
  }
  if (this->layer_param_.convolution_param().dis_mode() && this->gan_mode_ == 2) {
	update_weight = false;
  }
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1] && update_weight) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0] && update_weight) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  // update gan_mode_
  this->gan_mode_ = this->gan_mode_ == 2 ? 1 : this->gan_mode_ + 1;
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
