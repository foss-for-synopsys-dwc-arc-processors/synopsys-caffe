#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/mirror_pad_layer.hpp"

namespace caffe {
using namespace std;
template <typename Dtype>
void MirrorPadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  const MirrorPadParameter &mirror_pad_param =
      this->layer_param_.mirror_pad_param();
  constant_values_ = mirror_pad_param.constant_values();
  mode_ = mirror_pad_param.mode();
  paddings_.clear();
  std::copy(mirror_pad_param.paddings().begin(),
      mirror_pad_param.paddings().end(), std::back_inserter(paddings_));
  int pad_dim = paddings_.size();
  CHECK_EQ(pad_dim % 2, 0)
  << "Paddings for each dimension should have 2 values!";
  CHECK_EQ(pad_dim / 2, bottom[0]->num_axes())
  << "Paddings' num should be 2 times of bottom dimension!";
  // CHECK_LE(bottom[0]->num_axes(), 4) << "Not support more than 4D paddings!";
}

template <typename Dtype>
void MirrorPadLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  int num_top_axes = bottom[0]->num_axes();
  std::vector<int> shape(num_top_axes, 1);
  shape = bottom[0]->shape();
  for (int i = 0; i < num_top_axes; i++) {
    shape[i] = shape[i] + paddings_[2 * i] + paddings_[2 * i + 1];
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
inline vector<int>
MirrorPadLayer<Dtype>::indices(int offset, const vector<int> &shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for (int i = shape.size() - 1; i >= 0; i--) {
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int MirrorPadLayer<Dtype>::offset(const vector<int> &indices,
    const vector<int> &shape) const {
  int offset = 0;
  for (int i = 0; i < shape.size(); ++i) {
    offset *= shape[i];
    offset += indices[i];
  }
  return offset;
}

template <typename Dtype>
void MirrorPadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  auto bottom_shape = bottom[0]->shape();
  auto top_shape = top[0]->shape();
  int strides = bottom_shape.back();

  caffe_set(top[0]->count(), Dtype(constant_values_), top_data);

  for (int position = 0; position < bottom[0]->count() / strides; position++) {
    vector<int> coord_bottom = indices(position * strides, bottom_shape);
    vector<int> coord_pad(coord_bottom);
    for (int i = 0; i < top_shape.size(); i++)
      coord_pad[i] += paddings_[2 * i];
    int position_top = offset(coord_pad, top_shape);
    copy_n(bottom_data + position * strides, strides, top_data + position_top);
  }
  if (mode_ == "REFLECT") {
    strides = 1;
    for (int i = top_shape.size() - 1; i >= 0; i--) {
      int inner_strides = strides;
      strides *= top_shape[i];
      for (int position = 0; position < top[0]->count() / strides; position++) {
        for (int j = 1; j <= paddings_[2 * i]; j++) {
          copy_n(top_data + position*strides + inner_strides * (paddings_[2 * i] + j),
              inner_strides,
              top_data + position*strides + inner_strides * (paddings_[2 * i] - j));
        }
        for (int j = 1; j <= paddings_[2 * i + 1]; j++) {
          copy_n(top_data + position*strides + inner_strides * (bottom_shape[i] + paddings_[2 * i] - 1 - j),
              inner_strides,
              top_data + position*strides + inner_strides * (bottom_shape[i] + paddings_[2 * i] - 1 + j));
        }
      }
    }
  } else if (mode_ == "SYMMETRIC") {
    strides = 1;
    for (int i = top_shape.size() - 1; i >= 0; i--) {
      int inner_strides = strides;
      strides *= top_shape[i];
      for (int position = 0; position < top[0]->count() / strides; position++) {
        for (int j = 0; j < paddings_[2 * i]; j++) {
          copy_n(top_data + position*strides + inner_strides * (paddings_[2 * i] + j),
              inner_strides,
              top_data + position*strides + inner_strides * (paddings_[2 * i] - j - 1));
        }
        for (int j = 0; j < paddings_[2 * i + 1]; j++) {
          copy_n(top_data + position*strides + inner_strides * (bottom_shape[i] + paddings_[2 * i] - 1 - j),
              inner_strides,
              top_data + position*strides + inner_strides * (bottom_shape[i] + paddings_[2 * i] + j));
        }
      }
    }
  } else if (mode_ == "EDGE") {
    strides = 1;
    for (int i = top_shape.size() - 1; i >= 0; i--) {
      int inner_strides = strides;
      strides *= top_shape[i];
      for (int position = 0; position < top[0]->count() / strides; position++) {
        for (int j = 0; j < paddings_[2 * i]; j++) {
          copy_n(top_data + position*strides + inner_strides * (paddings_[2 * i]),
              inner_strides,
              top_data + position*strides + inner_strides * (paddings_[2 * i] - j - 1));
        }
        for (int j = 0; j < paddings_[2 * i + 1]; j++) {
          copy_n(top_data + position*strides + inner_strides * (bottom_shape[i] + paddings_[2 * i] - 1),
              inner_strides,
              top_data + position*strides + inner_strides * (bottom_shape[i] + paddings_[2 * i] + j));
        }
      }
    }
  }
}

INSTANTIATE_CLASS(MirrorPadLayer);
REGISTER_LAYER_CLASS(MirrorPad);

} // namespace caffe
