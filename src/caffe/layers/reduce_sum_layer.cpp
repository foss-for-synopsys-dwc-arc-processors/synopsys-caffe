#include <algorithm>
#include <vector>

#include "caffe/layers/reduce_sum_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void ReduceSumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  const ReduceSumParameter &reduce_sum_param =
      this->layer_param_.reduce_sum_param();
  reduce_sum_keepdims_ = reduce_sum_param.keepdims();
  reduce_sum_axis_.clear();
  std::copy(reduce_sum_param.axis().begin(), reduce_sum_param.axis().end(),
            std::back_inserter(reduce_sum_axis_));
  axis_dim_ = reduce_sum_axis_.size();
  CHECK_LE(axis_dim_, bottom[0]->num_axes())
      << "the dimension of axis should be less or equal than input dimension!";
  for (int i = 0; i < axis_dim_; ++i) {
    reduce_sum_axis_[i] = bottom[0]->CanonicalAxisIndex(reduce_sum_axis_[i]);
  }
  std::sort(reduce_sum_axis_.begin(), reduce_sum_axis_.end());
}

template <typename Dtype>
void ReduceSumLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const int num_axes = bottom[0]->num_axes();
  vector<int> top_shape = bottom[0]->shape();
  vector<int> bottom_shape = bottom[0]->shape();
  if (reduce_sum_keepdims_) {
    if (axis_dim_ != 0) {
      // has keepdims and axis
      for (int i = 0; i < axis_dim_; ++i) {
        top_shape[reduce_sum_axis_[i]] = 1;
      }
    } else {
      // has keepdims but no axis
      top_shape.assign(top_shape.size(), 1);
    }
  } else {
    if (axis_dim_ != 0) {
      // no keepdims but has axis
      for (int i = axis_dim_ - 1; i > -1; --i) {
        top_shape.erase(top_shape.begin() + reduce_sum_axis_[i]);
      }
    } else {
      // no axis and no keepdims
      top_shape.resize(0);
    }
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int>
ReduceSumLayer<Dtype>::indices(int offset, const vector<int> &shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for (int i = shape.size() - 1; i >= 0; i--) {
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int ReduceSumLayer<Dtype>::offset(const vector<Blob<Dtype> *> &bottom,
                                         const vector<int> &shape,
                                         const vector<int> &axis_ind,
                                         const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < axis_ind.size(); ++i) {
    offset += indices[i] * bottom[0]->count(axis_ind[i] + 1);
  }
  return offset;
}

template <typename Dtype>
void ReduceSumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const vector<int> bottom_shape = bottom[0]->shape();
  const int bottom_count = bottom[0]->count();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  if (axis_dim_ == 0 || axis_dim_ == bottom[0]->num_axes()) {
    // no axis, add all elements
    for (int i = 0; i < bottom_count; ++i) {
      top_data[0] += bottom_data[i];
    }
  } else {
    // has axis, add all elements in dim:reduce_sum_axis_
    vector<int> shape_out = bottom[0]->shape();
    vector<int> axis_out(bottom[0]->num_axes(), 0);
    for (int i = 0; i < axis_out.size(); ++i) {
      axis_out[i] = i;
    }
    for (int i = axis_dim_ - 1; i > -1; --i) {
      shape_out.erase(shape_out.begin() + reduce_sum_axis_[i]);
      axis_out.erase(axis_out.begin() + reduce_sum_axis_[i]);
    }

    vector<int> shape_in(reduce_sum_axis_.size(), 0);
    for (int i = 0; i < reduce_sum_axis_.size(); ++i) {
      shape_in[i] = bottom[0]->shape()[reduce_sum_axis_[i]];
    }

    for (int i = 0; i < top_count; ++i) {
      vector<int> ind_out = indices(i, shape_out);
      int offset_out = offset(bottom, bottom_shape, axis_out, ind_out);
      for (int j = 0; j < bottom_count / top_count; ++j) {
        vector<int> ind_in = indices(j, shape_in);
        int offset_in = offset(bottom, bottom_shape, reduce_sum_axis_, ind_in);
        int b_idx = offset_out + offset_in;
        top_data[i] += bottom_data[b_idx];
      }
    }
  }
}

INSTANTIATE_CLASS(ReduceSumLayer);
REGISTER_LAYER_CLASS(ReduceSum);

} // namespace caffe
