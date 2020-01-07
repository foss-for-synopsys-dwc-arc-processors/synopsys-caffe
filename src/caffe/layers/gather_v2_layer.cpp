#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/gather_v2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GatherV2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  // GatherV2 has 2 inputs: params, indices, 1 attribute:axis
  const GatherV2Parameter &gather_v2_param =
      this->layer_param_.gather_v2_param();
  gather_axis_ = bottom[0]->CanonicalAxisIndex(gather_v2_param.axis());
  indices_shape_ = bottom[1]->shape();
  CHECK_GE(bottom[0]->num_axes(), 1)
      << "the dimension of input should be larger than or equal to 1";
}

template <typename Dtype>
void GatherV2Layer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  // Initialize with the first blob
  // The result shape is params.shape[-1:axis] + indices.shape +
  // params.shape[axis + 0:].
  const int indices_dim_ = bottom[1]->num_axes();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.erase(top_shape.begin() + gather_axis_);
  top_shape.insert(top_shape.begin() + gather_axis_, indices_shape_.begin(),
                   indices_shape_.end());
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void GatherV2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  vector<int> bottom_shape = bottom[0]->shape();
  const int num_gather_ = bottom[0]->count(0, gather_axis_);
  const int gather_size_ = bottom[0]->count(gather_axis_ + 1);
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *indices_ = bottom[1]->cpu_data();
  // check indices_
  for (int i = 0; i < bottom[1]->count(); ++i) {
    CHECK_GE(indices_[i], 0)
        << "indices_ element with idx" << i << " is negative";
    CHECK_LT(indices_[i], bottom[0]->shape(gather_axis_))
        << "indices_ element with idx" << i << " is out of range "
        << bottom[0]->shape(gather_axis_);
  }
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int bottom_gather_axis = bottom[0]->shape(gather_axis_);
  int num = 0;
  for (int m = 0; m < num_gather_; ++m) {
    for (int n = 0; n < bottom[1]->count(); ++n) {
      const int top_offset = num * gather_size_;
      const int bottom_offset =
          (m * bottom_gather_axis + (int)indices_[n]) * gather_size_;
      caffe_copy(gather_size_, bottom_data + bottom_offset,
                 top_data + top_offset);
      num += 1;
    }
  }
}

INSTANTIATE_CLASS(GatherV2Layer);
REGISTER_LAYER_CLASS(GatherV2);

} // namespace caffe
