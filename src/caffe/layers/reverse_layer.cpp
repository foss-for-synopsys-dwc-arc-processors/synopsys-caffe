#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/reverse_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReverseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const ReverseParameter &reverse_param = this->layer_param_.reverse_param();
  axis_.clear();
  std::copy(reverse_param.axis().begin(), reverse_param.axis().end(),
            std::back_inserter(axis_));

  // check part
  const int num_axes = bottom[0]->num_axes();
  for (int i = 0; i < axis_.size(); ++i) {
    CHECK_GE(axis_[i], -num_axes)
        << "axis[i] should be in range [" << -num_axes << ", " << num_axes
        << "), but axis[" << i << "] = " << axis_[i];
    CHECK_LT(axis_[i], num_axes)
        << "axis[i] should be in range [" << -num_axes << ", " << num_axes
        << "), but axis[" << i << "] = " << axis_[i];
    axis_[i] = (axis_[i] >= 0) ? axis_[i] : axis_[i] + num_axes;
  }
  // sort axis
  std::sort(axis_.begin(), axis_.end());
}

template <typename Dtype>
void ReverseLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int>
ReverseLayer<Dtype>::indices(int offset, const vector<int> &shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for (int i = shape.size() - 1; i >= 0; i--) {
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int ReverseLayer<Dtype>::offset(const vector<Blob<Dtype> *> &bottom,
                                       const vector<int> &axis_ind,
                                       const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < axis_ind.size(); ++i) {
    offset += indices[i] * bottom[0]->count(axis_ind[i] + 1);
  }
  return offset;
}

template <typename Dtype>
inline vector<int>
ReverseLayer<Dtype>::reverse_indices(const vector<int> &indices,
                                     const vector<int> &shape) const {
  CHECK_EQ(indices.size(), shape.size())
      << "indices and shape should have same dimension, but got "
      << indices.size() << " and " << shape.size();
  vector<int> r_indices(indices.size(), 0);
  for (int i = 0; i < indices.size(); ++i) {
    r_indices[i] = shape[i] - indices[i] - 1;
  }
  return r_indices;
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  vector<int> bottom_shape = bottom[0]->shape();
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  vector<int> other_shape = bottom_shape;
  vector<int> other_axis(other_shape.size(), 0);
  vector<int> axis_shape(axis_.size(), 0);
  for (int i = 0; i < other_axis.size(); ++i) {
    other_axis[i] = i;
  }
  int axis_size = 1;
  for (int i = axis_.size() - 1; i > -1; --i) {
    other_shape.erase(other_shape.begin() + axis_[i]);
    other_axis.erase(other_axis.begin() + axis_[i]);
    axis_shape[i] = bottom_shape[axis_[i]];
    axis_size *= bottom_shape[axis_[i]];
  }
  const int other_count = bottom[0]->count() / axis_size;

  for (int m = 0; m < other_count; ++m) {
    vector<int> other_indices = indices(m, other_shape);
    int other_idx = offset(bottom, other_axis, other_indices);
    for (int n = 0; n < axis_size; ++n) {
      vector<int> baxis_indices = indices(n, axis_shape);
      vector<int> taxis_indices = reverse_indices(baxis_indices, axis_shape);
      int b_idx = other_idx + offset(bottom, axis_, baxis_indices);
      int t_idx = other_idx + offset(bottom, axis_, taxis_indices);
      top_data[t_idx] = bottom_data[b_idx];
    }
  }
}

INSTANTIATE_CLASS(ReverseLayer);
REGISTER_LAYER_CLASS(Reverse);

} // namespace caffe
