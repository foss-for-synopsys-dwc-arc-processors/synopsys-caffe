#include <algorithm>
#include <vector>

#include "caffe/layers/count_nonzero_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CountNonzeroLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  const CountNonzeroParameter &count_nonzero_param =
      this->layer_param_.count_nonzero_param();
  count_nonzero_keepdims_ = count_nonzero_param.keepdims();
  count_nonzero_axis_.clear();
  std::copy(count_nonzero_param.axis().begin(), count_nonzero_param.axis().end(),
            std::back_inserter(count_nonzero_axis_));
  axis_dim_ = count_nonzero_axis_.size();
  CHECK_LE(axis_dim_, bottom[0]->num_axes())
      << "the dimension of axis should be less or equal than input dimension!";
  for (int i = 0; i < axis_dim_; ++i) {
    count_nonzero_axis_[i] = bottom[0]->CanonicalAxisIndex(count_nonzero_axis_[i]);
  }
  std::sort(count_nonzero_axis_.begin(), count_nonzero_axis_.end());
}

template <typename Dtype>
void CountNonzeroLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  vector<int> bottom_shape = bottom[0]->shape();
  if (count_nonzero_keepdims_) {
    if (axis_dim_ != 0) {
      // has keepdims and axis
      for (int i = 0; i < axis_dim_; ++i) {
        top_shape[count_nonzero_axis_[i]] = 1;
      }
    } else {
      // has keepdims but no axis
      top_shape.assign(top_shape.size(), 1);
    }
  } else {
    if (axis_dim_ != 0) {
      // no keepdims but has axis
      for (int i = axis_dim_ - 1; i > -1; --i) {
        top_shape.erase(top_shape.begin() + count_nonzero_axis_[i]);
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
CountNonzeroLayer<Dtype>::indices(int offset, const vector<int> &shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for (int i = shape.size() - 1; i >= 0; i--) {
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int CountNonzeroLayer<Dtype>::offset(const vector<Blob<Dtype> *> &bottom,
                                         const vector<int> &axis_ind,
                                         const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < axis_ind.size(); ++i) {
    offset += indices[i] * bottom[0]->count(axis_ind[i] + 1);
  }
  return offset;
}

template <typename Dtype>
void CountNonzeroLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int bottom_count = bottom[0]->count();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  if (axis_dim_ == 0 || axis_dim_ == bottom[0]->num_axes()) {
    // no axis, add all elements
    for (int i = 0; i < bottom_count; ++i) {
      top_data[0] += int(bottom_data[i] != 0);
    }
  } else {
    // has axis, add all elements in dim:count_nonzero_axis_
    vector<int> shape_out = bottom[0]->shape();
    vector<int> axis_out(bottom[0]->num_axes(), 0);
    for (int i = 0; i < axis_out.size(); ++i) {
      axis_out[i] = i;
    }
    for (int i = axis_dim_ - 1; i > -1; --i) {
      shape_out.erase(shape_out.begin() + count_nonzero_axis_[i]);
      axis_out.erase(axis_out.begin() + count_nonzero_axis_[i]);
    }

    vector<int> shape_in(count_nonzero_axis_.size(), 0);
    for (int i = 0; i < count_nonzero_axis_.size(); ++i) {
      shape_in[i] = bottom[0]->shape()[count_nonzero_axis_[i]];
    }

    for (int i = 0; i < top_count; ++i) {
      vector<int> ind_out = indices(i, shape_out);
      int offset_out = offset(bottom, axis_out, ind_out);
      for (int j = 0; j < bottom_count / top_count; ++j) {
        vector<int> ind_in = indices(j, shape_in);
        int offset_in = offset(bottom, count_nonzero_axis_, ind_in);
        int b_idx = offset_out + offset_in;
        top_data[i] += int(bottom_data[b_idx] != 0);
      }
    }
  }
}

INSTANTIATE_CLASS(CountNonzeroLayer);
REGISTER_LAYER_CLASS(CountNonzero);

} // namespace caffe

