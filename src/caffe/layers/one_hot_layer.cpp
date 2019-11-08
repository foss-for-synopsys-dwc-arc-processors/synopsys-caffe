#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/one_hot_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OneHotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const OneHotParameter &one_hot_param = this->layer_param_.one_hot_param();
  depth = one_hot_param.depth();
  axis = one_hot_param.axis();
  on_value = one_hot_param.on_value();
  off_value = one_hot_param.off_value();
  num_axes = bottom[0]->num_axes();
  if (axis == -1) {
    axis = num_axes;
  }
  CHECK_LE(axis, num_axes) << "Expect axis to be -1 or between [0, "
                           << num_axes + 1 << "), but received " << axis;
  CHECK_GE(axis, 0) << "Expect axis to be -1 or between [0, " << num_axes + 1
                    << "), but received " << axis;
}

template <typename Dtype>
void OneHotLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  if (axis < num_axes) {
    top_shape.insert(top_shape.begin() + axis, depth);
  } else {
    top_shape.insert(top_shape.end(), depth);
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int> OneHotLayer<Dtype>::indices(int num,
                                               const vector<int> &shape) const {
  int r = num;
  vector<int> indices(shape.size(), 0);
  for (int i = shape.size() - 1; i > -1; --i) {
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int OneHotLayer<Dtype>::offset(const vector<Blob<Dtype> *> &top,
                                      const vector<int> &axis_ind,
                                      const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < indices.size(); ++i) {
    offset += indices[i] * top[0]->count(axis_ind[i] + 1);
  }
  return offset;
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(off_value), top_data);
  // if bottom data is scalar
  if (num_axes == 0) {
    int t_idx = bottom_data[0];
    if (t_idx >= 0 && t_idx < depth) {
      top_data[t_idx] = on_value;
    }
  }
  // bottom data is tensor
  else {
    vector<int> bottom_shape = bottom[0]->shape();
    vector<int> axis_ind(bottom_shape.size() + 1, 0);
    for (int i = 0; i < axis_ind.size(); ++i) {
      axis_ind[i] = i;
    }
    axis_ind.erase(axis_ind.begin() + axis);
    for (int i = 0; i < bottom[0]->count(); ++i) {
      vector<int> b_ind = indices(i, bottom_shape);
      int b_idx = offset(top, axis_ind, b_ind);
      if (bottom_data[i] >= 0 && bottom_data[i] < depth) {
        int t_idx = b_idx + bottom_data[i] * top[0]->count(axis + 1);
        top_data[t_idx] = on_value;
      }
    }
  }
}

INSTANTIATE_CLASS(OneHotLayer);
REGISTER_LAYER_CLASS(OneHot);

} // namespace caffe
