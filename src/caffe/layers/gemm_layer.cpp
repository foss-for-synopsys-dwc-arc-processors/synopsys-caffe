#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/gemm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GemmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  const GemmParameter &gemm_param = this->layer_param_.gemm_param();
  alpha_ = gemm_param.alpha();
  beta_ = gemm_param.beta();
  transa = gemm_param.transa();
  transb = gemm_param.transb();

  if (transa == 0) {
    M_ = bottom[0]->shape(0);
    K_ = bottom[0]->shape(1);
  } else {
    M_ = bottom[0]->shape(1);
    K_ = bottom[0]->shape(0);
  }

  if (transb == 0) {
    N_ = bottom[1]->shape(1);
    CHECK_EQ(bottom[1]->shape(0), K_) << "input A and input B have different K";
  } else {
    N_ = bottom[1]->shape(0);
    CHECK_EQ(bottom[1]->shape(1), K_) << "input A and input B have different K";
  }
}

template <typename Dtype>
void GemmLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                               const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = {M_, N_};
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void GemmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {

  const Dtype *bottom_A = bottom[0]->cpu_data();
  const Dtype *bottom_B = bottom[1]->cpu_data();
  const Dtype *bottom_C = bottom[2]->cpu_data();
  const vector<int> c_shape = bottom[2]->shape();
  Dtype *top_data = top[0]->mutable_cpu_data();

  if (bottom[2]->num_axes() == 2) {
    if (c_shape[0] == M_ && c_shape[1] == N_) {
      caffe_copy(top[0]->count(), bottom_C, top_data);
    } else if (c_shape[0] == M_ && c_shape[1] == 1) {
      for (int i = 0; i < M_; ++i) {
        caffe_set(N_, Dtype(bottom_C[i]), top_data + i * N_);
      }
    } else if (c_shape[0] == 1 && c_shape[1] == N_) {
      for (int i = 0; i < M_; ++i) {
        caffe_copy(N_, bottom_C, top_data + i * N_);
      }
    } else {
      std::cout
          << "input C should has shape: (M, N) or (M, 1) or (1, N) or but not"
          << std::endl;
    }
  } else if (bottom[2]->num_axes() == 1 && c_shape[0] == 1) {
    caffe_set(top[0]->count(), Dtype(bottom_C[0]), top_data);
  } else {
    std::cout << "input C has invalid shape for broadcast" << std::endl;
  }

  if (transa == 0 && transb == 0) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, K_, alpha_, bottom_A,
                   bottom_B, beta_, top_data);
  }
  if (transa == 0 && transb == 1) {
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, M_, N_, K_, alpha_, bottom_A,
                   bottom_B, beta_, top_data);
  }
  if (transa == 1 && transb == 0) {
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, M_, N_, K_, alpha_, bottom_A,
                   bottom_B, beta_, top_data);
  }
  if (transa == 1 && transb == 1) {
    caffe_cpu_gemm(CblasTrans, CblasTrans, M_, N_, K_, alpha_, bottom_A,
                   bottom_B, beta_, top_data);
  }
}

INSTANTIATE_CLASS(GemmLayer);
REGISTER_LAYER_CLASS(Gemm);

} // namespace caffe
