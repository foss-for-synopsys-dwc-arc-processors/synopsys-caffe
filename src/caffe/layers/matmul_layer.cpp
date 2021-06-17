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
  blob_shape_.clear();
  std::copy(matmul_param.blob_shape().begin(), matmul_param.blob_shape().end(),
            std::back_inserter(blob_shape_));

  if (bottom.size() == 1 && this->blobs_.size() != 1 &&
      blob_shape_.size() != 0) {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(blob_shape_));
    // initialize blobs_ value with  0.
    caffe_set(this->blobs_[0]->count(), Dtype(0),
              this->blobs_[0]->mutable_cpu_data());
  }
  Blob<Dtype> *inputs1 =
      (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  num_axes = bottom[0]->num_axes();

  CHECK_GE(bottom[0]->num_axes(), inputs1->num_axes())
      << "input a and input b should have same dimension or dim(a) > dim(b)!!";

  if (bottom[0]->num_axes() == inputs1->num_axes()) {
    M = transpose_a ? bottom[0]->shape(num_axes - 1)
                    : bottom[0]->shape(num_axes - 2);
    N = transpose_b ? inputs1->shape(num_axes - 2)
                    : inputs1->shape(num_axes - 1);
    K = transpose_a ? bottom[0]->shape(num_axes - 2)
                    : bottom[0]->shape(num_axes - 1);
    if (transpose_b) {
      CHECK_EQ(K, inputs1->shape(num_axes - 1))
          << "input a and input b have incompatible shapes! ";
    } else {
      CHECK_EQ(K, inputs1->shape(num_axes - 2))
          << "input a and input b have incompatible shapes! ";
    }
    for (int i = 0; i < num_axes - 2; i++) {
      CHECK_EQ(bottom[0]->shape(i), inputs1->shape(i))
          << "inputs should have same shape except in last two dimensions, but "
             "in dimension "
          << i << ", the two inputs have different shape!";
    }
  } else {
    int axes1 = bottom[0]->num_axes();
    int axes2 = inputs1->num_axes();
    K = bottom[0]->shape(axes1 - 1);
    M = bottom[0]->count() / K;
    N = inputs1->shape(axes2 - 1);
    CHECK_GE(axes2, 2) << "If dim(a) > dim(b), dim(b) should be 2!!";
    CHECK_EQ(K, inputs1->shape(axes2 - 2))
        << "input a and input b have incompatible shapes! ";
  }
}

template <typename Dtype>
void MatMulLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[num_axes - 1] = N;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MatMulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  Blob<Dtype> *inputs1 =
      (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  const Dtype *bottom_data0 = bottom[0]->cpu_data();
  const Dtype *bottom_data1 = inputs1->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  const int batch_size = bottom[0]->count() / (M * K);
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
