#include <algorithm>
#include <vector>

#include "caffe/layers/layer_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LayerNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  LayerNormParameter param = this->layer_param_.layer_norm_param();
  eps_ = param.eps();
  elementwise_affine_ = param.elementwise_affine();
  if (elementwise_affine_ && this->blobs_.size() != 2) {
    this->blobs_.resize(2);
    vector<int> param_shape = bottom[0]->shape();
    param_shape.erase(param_shape.begin());
    this->blobs_[0].reset(new Blob<Dtype>(param_shape));
    this->blobs_[1].reset(new Blob<Dtype>(param_shape));
    // initialize blobs_ value with 1. and 0.
    caffe_set(this->blobs_[0]->count(), Dtype(1),
              this->blobs_[0]->mutable_cpu_data());
    caffe_set(this->blobs_[1]->count(), Dtype(0),
              this->blobs_[1]->mutable_cpu_data());
  }
}

template <typename Dtype>
void LayerNormLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LayerNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  vector<int> bottom_shape = bottom[0]->shape();
  const int batch_ = bottom[0]->shape(0);
  const int flat_num = bottom[0]->count() / batch_;

  vector<int> sz;
  sz.push_back(batch_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);

  // set initial value
  if (batch_sum_multiplier_.num_axes() == 0 ||
      batch_sum_multiplier_.shape(0) != batch_) {
    sz[0] = batch_;
    batch_sum_multiplier_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
              batch_sum_multiplier_.mutable_cpu_data());
  }
  if (flat_sum_multiplier_.num_axes() == 0 ||
      flat_sum_multiplier_.shape(0) != flat_num) {
    sz[0] = flat_num;
    flat_sum_multiplier_.Reshape(sz);
    caffe_set(flat_sum_multiplier_.count(), Dtype(1),
              flat_sum_multiplier_.mutable_cpu_data());
  }

  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  // compute mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, batch_, flat_num, 1. / flat_num,
                        bottom_data, flat_sum_multiplier_.cpu_data(), 0.,
                        mean_.mutable_cpu_data());
  // subtract mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_, flat_num, 1, -1,
                        mean_.cpu_data(), flat_sum_multiplier_.cpu_data(), 1.,
                        top_data);
  // compute variance using var(X) = E((X-E(X))^2)
  caffe_sqr<Dtype>(top[0]->count(), top_data, temp_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, batch_, flat_num, 1. / flat_num,
                        temp_.cpu_data(), flat_sum_multiplier_.cpu_data(), 0.,
                        variance_.mutable_cpu_data());
  // add eps then sqrt
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_sqrt(variance_.count(), variance_.cpu_data(),
             variance_.mutable_cpu_data());
  // divide normalized variance
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_, flat_num, 1, 1.,
                        variance_.cpu_data(), flat_sum_multiplier_.cpu_data(),
                        0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  // elementwise_affine
  if (elementwise_affine_) {
    const Dtype *affine_scale_ = this->blobs_[0]->cpu_data();
    const Dtype *affine_bias_ = this->blobs_[1]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_, flat_num, 1, 1.,
                          batch_sum_multiplier_.cpu_data(), affine_scale_, 0.,
                          temp_.mutable_cpu_data());
    caffe_mul(temp_.count(), top_data, temp_.cpu_data(), top_data);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_, flat_num, 1, 1.,
                          batch_sum_multiplier_.cpu_data(), affine_bias_, 0.,
                          temp_.mutable_cpu_data());
    caffe_add(temp_.count(), top_data, temp_.cpu_data(), top_data);
  }
}

INSTANTIATE_CLASS(LayerNormLayer);
REGISTER_LAYER_CLASS(LayerNorm);

} // namespace caffe
