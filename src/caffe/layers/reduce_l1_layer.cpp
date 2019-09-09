#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/reduce_l1_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReduceL1Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const ReduceL1Parameter &reduce_l1_param =
      this->layer_param_.reduce_l1_param();
  axes_.clear();
  std::copy(reduce_l1_param.axes().begin(), reduce_l1_param.axes().end(),
            std::back_inserter(axes_));
  axes_dim_ = axes_.size();
  keepdims_ = reduce_l1_param.keepdims();
  // sort axes in order
  CHECK_LE(axes_dim_, bottom[0]->num_axes())
      << "the dimension of axes should be less or equal than input dimension!";
  for (int i = 0; i < axes_dim_; ++i) {
    axes_[i] = bottom[0]->CanonicalAxisIndex(axes_[i]);
  }
  std::sort(axes_.begin(), axes_.end());
}

template <typename Dtype>
void ReduceL1Layer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape = bottom_shape;
  if (keepdims_ == 1) {
    // has keepdims and axes
    if (axes_dim_ != 0) {
      for (int i = 0; i < axes_.size(); i++) {
        top_shape[axes_[i]] = 1;
      }
    }
    // has keepdims but no axes
    else {
      top_shape.assign(top_shape.size(), 1);
    }
  } else {
    // no keepdims but axes
    if (axes_dim_ != 0) {
      for (int i = axes_.size() - 1; i > -1; --i) {
        top_shape.erase(top_shape.begin() + axes_[i]);
      }
    }
    // no keepdims and no axes
    else {
      top_shape.resize(0);
    }
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int>
ReduceL1Layer<Dtype>::indices(int offset, const vector<int> &shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for (int i = shape.size() - 1; i >= 0; i--) {
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int ReduceL1Layer<Dtype>::offset(const vector<Blob<Dtype> *> &bottom,
                                        const vector<int> &axes_ind,
                                        const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < axes_ind.size(); ++i) {
    offset += indices[i] * bottom[0]->count(axes_ind[i] + 1);
  }
  return offset;
}

template <typename Dtype>
void ReduceL1Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int bottom_count = bottom[0]->count();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  if (axes_dim_ == 0 || axes_dim_ == bottom[0]->num_axes()) {
    // no axes, add all elements
    for (int i = 0; i < bottom_count; ++i) {
      top_data[0] += std::abs(bottom_data[i]);
    }
  } else {
    // has axes, add all elements in dim: axes_
    vector<int> shape_out = bottom[0]->shape();
    vector<int> axes_out(bottom[0]->num_axes(), 0);
    for (int i = 0; i < axes_out.size(); ++i) {
      axes_out[i] = i;
    }
    for (int i = axes_dim_ - 1; i > -1; --i) {
      shape_out.erase(shape_out.begin() + axes_[i]);
      axes_out.erase(axes_out.begin() + axes_[i]);
    }
    vector<int> shape_in(axes_.size(), 0);
    for (int i = 0; i < axes_.size(); ++i) {
      shape_in[i] = bottom[0]->shape()[axes_[i]];
    }
    for (int i = 0; i < top_count; ++i) {
      vector<int> ind_out = indices(i, shape_out);
      int offset_out = offset(bottom, axes_out, ind_out);
      for (int j = 0; j < bottom_count / top_count; ++j) {
        vector<int> ind_in = indices(j, shape_in);
        int offset_in = offset(bottom, axes_, ind_in);
        int b_idx = offset_out + offset_in;
        top_data[i] += std::abs(bottom_data[b_idx]);
      }
    }
  }
}

INSTANTIATE_CLASS(ReduceL1Layer);
REGISTER_LAYER_CLASS(ReduceL1);

} // namespace caffe
