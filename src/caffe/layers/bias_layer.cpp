#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // bias is a learned parameter; initialize it
    const BiasParameter& param = this->layer_param_.bias_param();
    const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
          << "bias blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> bias_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(param.filler()));
    filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // <--- CUSTOMIZATION
  const BiasParameter& param = this->layer_param_.bias_param();
  input_scale_ = param.input_scale();
  output_scale_ = param.output_scale();;
  bias_scale_ = param.bias_scale();
  input_zero_point_ = param.input_zero_point();
  output_zero_point_ = param.output_zero_point();
  bias_zero_point_ = param.bias_zero_point();
  saturate_ = param.saturate();
  quantize_method_ = param.quantize_method();
  // ---> CUSTOMIZATION
}

template <typename Dtype>
void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BiasParameter& param = this->layer_param_.bias_param();
  Blob<Dtype>* bias = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis == 0 in special case where bias is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis == 0 and (therefore) outer_dim_ == 1.
  const int axis = (bias->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis + bias->num_axes())
      << "bias blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis;
  for (int i = 0; i < bias->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis + i), bias->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis + i
        << ") and bias->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis);
  bias_dim_ = bias->count();
  inner_dim_ = bottom[0]->count(axis + bias->num_axes());
  dim_ = bias_dim_ * inner_dim_;
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
  bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
  if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)) {
    caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int bias_count = ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->count();
  Dtype* bias_mut = ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->mutable_cpu_data(); // bias_mutatble
  const int input_count = bottom[0]->count();
  Dtype* input_mut = bottom[0]->mutable_cpu_data(); // input_mutatble
  const bool quant_in = (input_scale_ != Dtype(1)) || (input_zero_point_ != 0);
  const bool quant_bias = (bias_scale_ != Dtype(1)) || (bias_zero_point_ != 0);
  const bool quant_out = (output_scale_ != Dtype(1)) || (output_zero_point_ != 0);
  // if (quantize_method_ == ?)
  // Currently, only onnx has bias_layer; the quantized computation is for onnx.
  if (quant_in) caffe_cpu_dequantize(input_count, input_mut, input_scale_, input_zero_point_);
  if (quant_bias) caffe_cpu_dequantize(bias_count, bias_mut, bias_scale_, bias_zero_point_);
  // https://github.com/microsoft/onnxruntime/blob/6334c292408a1df4d19835a1edd1d563d932880b/onnxruntime/test/mlas/unittest/test_qlinear_binaryop.cpp#L43
  const Dtype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (bottom[0] != top[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  for (int n = 0; n < outer_dim_; ++n) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_,
        inner_dim_, 1, Dtype(1), bias_data,
        bias_multiplier_.cpu_data(), Dtype(1), top_data);
    top_data += dim_;
  }
  if (quant_out) {
    const int count_t = top[0]->count();
    top_data = top[0]->mutable_cpu_data();
    caffe_cpu_quantize(count_t, top_data, output_scale_, output_zero_point_);
    caffe_cpu_saturate(count_t, top_data, saturate_); // if None nothing happens
  }
  if (quant_in) caffe_cpu_quantize(input_count, input_mut, input_scale_, input_zero_point_);
  if (quant_bias) caffe_cpu_quantize(bias_count, bias_mut, bias_scale_, bias_zero_point_);
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_cpu_diff();
    bool accum = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_cpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
          top_diff, bias_multiplier_.cpu_data(), Dtype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BiasLayer);
#endif

INSTANTIATE_CLASS(BiasLayer);
REGISTER_LAYER_CLASS(Bias);

}  // namespace caffe
