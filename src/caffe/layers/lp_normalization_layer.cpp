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
  axis_ = (lp_normalization_param.axis() < 0)
              ? lp_normalization_param.axis() + bottom[0]->num_axes()
              : lp_normalization_param.axis();
  p_ = lp_normalization_param.p();

  CHECK_LT(axis_, bottom[0]->num_axes())
      << "the dimension of axis should be less than input dimension!";
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
  shape_out.erase(shape_out.begin() + axis_);
  axis_out.erase(axis_out.begin() + axis_);

  const int shape_in = bottom[0]->shape(axis_);
  const int out_count = bottom_count / shape_in;

  for (int i = 0; i < out_count; ++i) {
    vector<int> ind_out = indices(i, shape_out);
    int offset_out = offset(bottom, axis_out, ind_out);
    Dtype nsum = 0;

    for (int j = 0; j < shape_in; ++j) {
      int offset_in = j * bottom[0]->count(axis_ + 1);
      int b_idx = offset_out + offset_in;
      if (p_ == 1) {
        nsum += std::abs(bottom_data[b_idx]);
      } else {
        CHECK_EQ(p_, 2) << "parameter p should be 1 or 2, not other numbers!!";
        nsum += std::pow(bottom_data[b_idx], 2);
      }
    }
    if (p_ == 2) {
      nsum = std::pow(nsum, 0.5);
    }

    for (int j = 0; j < shape_in; ++j) {
      int offset_in = j * bottom[0]->count(axis_ + 1);
      int b_idx = offset_out + offset_in;
      top_data[b_idx] = bottom_data[b_idx] / nsum;
    }
  }
}

INSTANTIATE_CLASS(LpNormalizationLayer);
REGISTER_LAYER_CLASS(LpNormalization);

} // namespace caffe
