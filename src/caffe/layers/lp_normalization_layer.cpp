#include <algorithm>
#include <vector>

#include "caffe/layers/lp_normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void LpNormalizationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const LpNormalizationParameter &lp_normalization_param =
      this->layer_param_.lp_normalization_param();
  p_ = lp_normalization_param.p();
  axis_.clear();
  std::copy(lp_normalization_param.axis().begin(),
            lp_normalization_param.axis().end(), std::back_inserter(axis_));
  axis_dim_ = axis_.size();
  CHECK_LE(axis_dim_, bottom[0]->num_axes())
      << "the dimension of axis should be less or equal than input dimension!";
  for (int i = 0; i < axis_dim_; ++i) {
    axis_[i] = bottom[0]->CanonicalAxisIndex(axis_[i]);
  }
  std::sort(axis_.begin(), axis_.end());
}

template <typename Dtype>
void LpNormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int>
LpNormalizationLayer<Dtype>::indices(int offset,
                                     const vector<int> &shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for (int i = shape.size() - 1; i >= 0; i--) {
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int
LpNormalizationLayer<Dtype>::offset(const vector<Blob<Dtype> *> &bottom,
                                    const vector<int> &axis_ind,
                                    const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < axis_ind.size(); ++i) {
    offset += indices[i] * bottom[0]->count(axis_ind[i] + 1);
  }
  return offset;
}

template <typename Dtype>
void LpNormalizationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int bottom_count = bottom[0]->count();
  caffe_set(bottom_count, Dtype(0), top_data);

  vector<int> shape_out = bottom[0]->shape();
  vector<int> axis_out(bottom[0]->num_axes(), 0);
  for (int i = 0; i < axis_out.size(); ++i) {
    axis_out[i] = i;
  }
  for (int i = axis_dim_ - 1; i > -1; --i) {
    shape_out.erase(shape_out.begin() + axis_[i]);
    axis_out.erase(axis_out.begin() + axis_[i]);
  }

  vector<int> shape_in(axis_.size(), 0);
  for (int i = 0; i < axis_.size(); ++i) {
    shape_in[i] = bottom[0]->shape()[axis_[i]];
  }

  int in_count = 1;
  for (int i = 0; i < shape_in.size(); ++i) {
    in_count *= shape_in[i];
  }
  int out_count = bottom_count / in_count;

  for (int i = 0; i < out_count; ++i) {
    vector<int> ind_out = indices(i, shape_out);
    int offset_out = offset(bottom, axis_out, ind_out);
    Dtype nsum = 0;
    for (int j = 0; j < in_count; ++j) {
      vector<int> ind_in = indices(j, shape_in);
      int offset_in = offset(bottom, axis_, ind_in);
      int b_idx = offset_out + offset_in;
      nsum += std::pow(std::abs(bottom_data[b_idx]), p_);
    }
    const float rp = 1.0 / p_;
    nsum = std::pow(nsum, rp);

    for (int j = 0; j < in_count; ++j) {
      vector<int> ind_in = indices(j, shape_in);
      int offset_in = offset(bottom, axis_, ind_in);
      int b_idx = offset_out + offset_in;
      top_data[b_idx] = bottom_data[b_idx] / nsum;
    }
  }
}

INSTANTIATE_CLASS(LpNormalizationLayer);
REGISTER_LAYER_CLASS(LpNormalization);

} // namespace caffe
