#include <vector>

#include "caffe/layers/conv_depthwise_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionDepthwiseLayer<Dtype>::compute_output_shape() {
  /* Blob<int> kernel_shape_, stride_, pad_, ..., 
       num_spatial_axes_, ... all declared in base_conv_layer.hpp */
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
void ConvolutionDepthwiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) { // num_ seems to be 'batch size'
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionDepthwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();  // the gradient dLoss/dY
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
        // gradient w.r.t. weight. Note that we will accumulate diffs.  // dLoss/dW = dLoss/dY * dY/dW
        if (this->param_propagate_down_[0]) {
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
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionDepthwiseLayer);
#endif

INSTANTIATE_CLASS(ConvolutionDepthwiseLayer);
REGISTER_LAYER_CLASS(ConvolutionDepthwise);

}  // namespace caffe