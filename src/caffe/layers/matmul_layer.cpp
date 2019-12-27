#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/matmul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatMulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const MatMulParameter &matmul_param = this->layer_param_.matmul_param();
  transpose_a = matmul_param.transpose_a();
  transpose_b = matmul_param.transpose_b();

  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
      << "input a and input b should have same dimension!!";
  num_axes = bottom[0]->num_axes();
  M = transpose_a ? bottom[0]->shape(num_axes - 1)
                  : bottom[0]->shape(num_axes - 2);
  N = transpose_b ? bottom[1]->shape(num_axes - 2)
                  : bottom[1]->shape(num_axes - 1);
  K = transpose_a ? bottom[0]->shape(num_axes - 2)
                  : bottom[0]->shape(num_axes - 1);
  if (transpose_b) {
    CHECK_EQ(K, bottom[1]->shape(num_axes - 1))
        << "input a and input b have incompatible shapes! ";
  } else {
    CHECK_EQ(K, bottom[1]->shape(num_axes - 2))
        << "input a and input b have incompatible shapes! ";
  }
  for (int i = 0; i < num_axes - 2; i++) {
    CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i))
        << "inputs should have same shape except in last two dimensions, but "
           "in dimension "
        << i << ", the two inputs have different shape!";
  }
}

template <typename Dtype>
void MatMulLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {

  vector<int> top_shape = bottom[0]->shape();
  top_shape[num_axes - 2] = M;
  top_shape[num_axes - 1] = N;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MatMulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {

  const Dtype *bottom_data0 = bottom[0]->cpu_data();
  const Dtype *bottom_data1 = bottom[1]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  const int batch_size = bottom[0]->count(0, num_axes - 2);

  for (int i = 0; i < batch_size; ++i) {
    int b_idx0 = i * M * K;
    int b_idx1 = i * K * N;
    int t_idx = i * M * N;
    caffe_cpu_gemm<Dtype>(transpose_a ? CblasTrans : CblasNoTrans,
                          transpose_b ? CblasTrans : CblasNoTrans, M, N, K,
                          (Dtype)1., bottom_data0 + b_idx0,
                          bottom_data1 + b_idx1, (Dtype)0., top_data + t_idx);
  }
}

INSTANTIATE_CLASS(MatMulLayer);
REGISTER_LAYER_CLASS(MatMul);

} // namespace caffe
