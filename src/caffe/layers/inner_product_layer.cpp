#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define W this->blobs_[0]
#define B this->blobs_[1]

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  update_weight_ = !this->layer_param_.inner_product_param().weight_fixed();
  N_ = num_output;
  gan_mode_ = 1;
  input_scale_ = this->layer_param_.inner_product_param().input_scale();  //CUSTOMIZATION
  output_scale_ = this->layer_param_.inner_product_param().output_scale();  //CUSTOMIZATION
  weight_scale_ = this->layer_param_.inner_product_param().weight_scale();  //CUSTOMIZATION
  input_zero_point_ = this->layer_param_.inner_product_param().input_zero_point();  //CUSTOMIZATION
  output_zero_point_ = this->layer_param_.inner_product_param().output_zero_point();  //CUSTOMIZATION
  weight_zero_point_ = this->layer_param_.inner_product_param().weight_zero_point();  //CUSTOMIZATION
  saturate_ = this->layer_param_.inner_product_param().saturate();  //CUSTOMIZATION
  quantize_method_ = this->layer_param_.inner_product_param().quantize_method();  //CUSTOMIZATION
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

#include "conv_layer.ev.inc"
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const bool shift_input = (input_zero_point_ != 0);
  const bool shift_weight = (weight_zero_point_ != 0);
  const bool scale_output = (input_scale_ != Dtype(1.0) || weight_scale_ != Dtype(1.0) ||
                              output_scale_ != Dtype(1.0));
  const bool shift_output = (output_zero_point_ != 0);
  if (shift_weight) { // shift the quantized weight
    caffe_add_scalar<Dtype>(W->count(), Dtype(-weight_zero_point_), W->mutable_cpu_data());
  }
  if (shift_input) {
    caffe_add_scalar<Dtype>(bottom[0]->count(),
        Dtype(-input_zero_point_), bottom[0]->mutable_cpu_data());
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
  const int count_t = top[0]->count();
  if (scale_output) {
    // refer out_multiplier to https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/kernel_util.cc#L41
    double out_scal = (double)input_scale_ * weight_scale_;
    out_scal /= output_scale_;
    if (quantize_method_ == InnerProductParameter_QuantizeMethod_tflite) {
      caffe_cpu_scale_double_round(count_t, out_scal, top_data);
    } else { // quantize_method_ == PoolingParameter_QuantizeMethod_ONNX
      for (int k = 0; k < count_t; ++k) {
        top_data[k] = std::rint(top_data[k] * out_scal);
      }
    }
  }
  if (shift_output) {
    caffe_add_scalar<Dtype>(count_t, Dtype(output_zero_point_), top_data);
  }
  caffe_cpu_saturate(count_t, top_data, saturate_); // if None nothing happens

  if (shift_input) { // shift the quantized input blob back to correct range
    caffe_add_scalar<Dtype>(bottom[0]->count(),
        Dtype(input_zero_point_), bottom[0]->mutable_cpu_data());
  }
  // shift quantized weight/bias back to correct range
  if (shift_weight) {
    caffe_add_scalar<Dtype>(W->count(), Dtype(weight_zero_point_), W->mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
	update_weight_ = true;
    if (this->layer_param_.inner_product_param().gen_mode() && gan_mode_ != 2) {
      update_weight_ = false;
    }
    if (this->layer_param_.inner_product_param().dis_mode() && gan_mode_ == 2) {
      update_weight_ = false;
    }
    // Gradient with respect to weight
    if (transpose_) {
      if (update_weight_) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
      }
    } else {
      if (update_weight_) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
      }
    }
  }
  if (bias_term_ && this->param_propagate_down_[1] && update_weight_) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
  // update gan_mode_
  gan_mode_ = gan_mode_ == 2 ? 1 : gan_mode_ + 1;
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
