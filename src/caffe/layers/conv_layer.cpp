#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define W this->blobs_[0]
#define B this->blobs_[1]

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
  const double input_scale = this->input_scale_;
  const double output_scale = this->output_scale_;
  const double weight_scale = this->weight_scale_;
  // bias_scale = input_scale * weight_scale
  const int input_zero_point = this->input_zero_point_;
  const int output_zero_point = this->output_zero_point_;
  const int weight_zero_point = this->weight_zero_point_;
  const Dtype saturate = this->saturate_;
  const int quantize_method = this->quantize_method_;
  // weight/bias scale will retrieve the default values if per-channel is set
  const bool per_channel_scale_weight = this->per_channel_scale_weight_;
  /*** Quantization Computation
    (1) shift input/weight/bias w.r.t corresponding zero_point
    (2) compute Convolution+Bias on the integer value range
    (3) scale the output by input_scale*weight_scale/output_scale, and
    (4) shift the output by output_zero_point
  *Assumption is that bias_scale = input_scale*weight_scale
  For a floating-value model, only (2) is computed with floating values
  ***/
  const bool shift_input = (input_zero_point != 0);
  const bool shift_weight = (weight_zero_point != 0);
  const bool scale_output = (input_scale != Dtype(1.0) || weight_scale != Dtype(1.0) ||
                             output_scale != Dtype(1.0)) || per_channel_scale_weight;
  const bool shift_output = (output_zero_point != 0);
  const bool is_depthwise = (this->group_ == this->num_output_);
  const bool is_pointwise = (W->count(2) == 1); // NCHW, count(HW) == 1

  const int quant_num_ch = per_channel_scale_weight ? this->num_output_ : 1;
  const Dtype* weight_scale_data = per_channel_scale_weight ? this->blobs_[2]->cpu_data() : NULL;
  const Dtype* weight_zero_point_data = per_channel_scale_weight ? this->blobs_[3]->cpu_data() : NULL;

  if (shift_weight) { // shift the quantized weight
    caffe_add_scalar<Dtype>(W->count(), Dtype(-weight_zero_point), W->mutable_cpu_data());
  } else if(per_channel_scale_weight) {
    const int slice = W->count() / quant_num_ch;
    Dtype* weight_mutable = W->mutable_cpu_data();
    for (int i = 0; i < quant_num_ch; ++i) {
      caffe_add_scalar<Dtype>(slice, -weight_zero_point_data[i], weight_mutable);
      weight_mutable += slice;
    }
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
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }

    const int count_t = top[i]->count();
    if (scale_output) {
      if (quantize_method == ConvolutionParameter_QuantizeMethod_TensorFlowLite) {
        if (per_channel_scale_weight) {
          const int slice = count_t / quant_num_ch;
          Dtype* top_mutable = top[i]->mutable_cpu_data();
          for (int j = 0; j < quant_num_ch; ++j) {
            double out_scal = input_scale * (double)weight_scale_data[j] / output_scale;
            int q_shift;
            int q_scal = tfl_QuantizeMultiplier(out_scal, &q_shift);

            if (is_depthwise) {
              // double rounding
              MultiplyByQuantizedMultiplierVR(slice, top_mutable, q_scal, q_shift, 2);
            } else {
              // It is found at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantized_conv_ops.cc
              // single rounding
              MultiplyByQuantizedMultiplierVR(slice, top_mutable, q_scal, q_shift, 1);
            }
            top_mutable += slice;
          }
        } else {
          // refer out_multiplier to https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/kernel_util.cc#L41
          double out_scal = (double)input_scale * weight_scale;
          out_scal /= output_scale;
          int q_shift;
          int q_scal = tfl_QuantizeMultiplier(out_scal, &q_shift);
          //caffe_cpu_scale_double_round(count_t, out_scal, top_data);
          //MultiplyByQuantizedMultiplierVR(count_t, top_data, q_scal, q_shift, 2);
          if (is_depthwise || is_pointwise) {
            double ref_scal = (float)input_scale * (float)weight_scale;
            ref_scal /= (float) output_scale; // break the scale into two lines to reproduce the error
            q_scal = tfl_QuantizeMultiplier(ref_scal, &q_shift);
            // ref_scale to reproduce error https://github.com/tensorflow/tensorflow/issues/23800
            // this error is observed in uint8 models, with input=[0,255]. TF version=2.5.0rc0
            MultiplyByQuantizedMultiplierVR(count_t, top_data, q_scal, q_shift, 2);
          } else {
            MultiplyByQuantizedMultiplierVR(count_t, top_data, q_scal, q_shift, 2);
          }
        }
      }
      else { // quantize_method_ == PoolingParameter_QuantizeMethod_ONNX
        float onnx_scale = (float) input_scale * (float) weight_scale / (float) output_scale;
        for (int k = 0; k < count_t; ++k) {
          top_data[k] = std::rint(top_data[k] * onnx_scale);
        }
      }
    }

    if (shift_output) {
      caffe_add_scalar<Dtype>(count_t, Dtype(output_zero_point), top_data);
    }

    caffe_cpu_saturate(count_t, top_data, saturate); // if None nothing happens

    if (shift_input) { // shift the quantized input blob back to correct range
      caffe_add_scalar<Dtype>(bottom[i]->count(),
        Dtype(input_zero_point), bottom[i]->mutable_cpu_data());
    }
  }
  // shift quantized weight/bias back to correct range
  if (shift_weight) {
    caffe_add_scalar<Dtype>(W->count(), Dtype(weight_zero_point), W->mutable_cpu_data());
  } else if(per_channel_scale_weight) {
    const int slice = W->count() / quant_num_ch;
    Dtype* weight_mutable = W->mutable_cpu_data();
    for (int i = 0; i < quant_num_ch; ++i) {
      caffe_add_scalar<Dtype>(slice, weight_zero_point_data[i], weight_mutable);
      weight_mutable += slice;
    }
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
