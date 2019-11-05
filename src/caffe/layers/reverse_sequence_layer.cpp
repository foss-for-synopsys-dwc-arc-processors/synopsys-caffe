#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/reverse_sequence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReverseSequenceLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const ReverseSequenceParameter &reverse_sequence_param =
      this->layer_param_.reverse_sequence_param();
  seq_lengths_.clear();
  std::copy(reverse_sequence_param.seq_lengths().begin(),
            reverse_sequence_param.seq_lengths().end(),
            std::back_inserter(seq_lengths_));
  seq_axis_ = reverse_sequence_param.seq_axis();
  batch_axis_ = reverse_sequence_param.batch_axis();
  // check part
  const int num_axes = bottom[0]->num_axes();
  CHECK_GE(batch_axis_, 0) << "batch_axis is less than 0!!";
  CHECK_LT(batch_axis_, num_axes) << "batch_axis must be < " << num_axes;
  CHECK_GE(seq_axis_, 0) << "seq_axis is less than 0!!";
  CHECK_LT(seq_axis_, num_axes) << "seq_axis must be < " << num_axes;
  CHECK_NE(batch_axis_, seq_axis_) << "batch_axis should not equal to seq_axis";
  CHECK_EQ(seq_lengths_.size(), bottom[0]->shape(batch_axis_))
      << "seq_lengths should equal to input.dims(batch_axis), but are "
      << seq_lengths_.size() << " and " << bottom[0]->shape(batch_axis_);
  for (int i = 0; i < seq_lengths_.size(); ++i) {
    CHECK_GE(seq_lengths_[i], 0)
        << "seq_lengths[" << i << "] should not be negative";
    CHECK_LE(seq_lengths_[i], bottom[0]->shape(seq_axis_))
        << "seq_lengths[i] should not be larger than input.dims(seq_axis), but "
           "get seq_lengths["
        << i << "] = " << seq_lengths_[i] << " > "
        << bottom[0]->shape(seq_axis_);
  }
}

template <typename Dtype>
void ReverseSequenceLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int>
ReverseSequenceLayer<Dtype>::indices(int offset,
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
ReverseSequenceLayer<Dtype>::offset(const vector<Blob<Dtype> *> &bottom,
                                    const vector<int> &axis_ind,
                                    const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < axis_ind.size(); ++i) {
    offset += indices[i] * bottom[0]->count(axis_ind[i] + 1);
  }
  return offset;
}

template <typename Dtype>
void ReverseSequenceLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  vector<int> bottom_shape = bottom[0]->shape();
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int batch_size = bottom[0]->shape(batch_axis_);
  const int seq_size = bottom[0]->shape(seq_axis_);
  const int other_count = bottom[0]->count() / (batch_size * seq_size);
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  vector<int> other_shape = bottom_shape;
  vector<int> other_axis(other_shape.size(), 0);
  for (int i = 0; i < other_axis.size(); ++i) {
    other_axis[i] = i;
  }
  other_shape.erase(other_shape.begin() + std::max(seq_axis_, batch_axis_));
  other_shape.erase(other_shape.begin() + std::min(seq_axis_, batch_axis_));
  other_axis.erase(other_axis.begin() + std::max(seq_axis_, batch_axis_));
  other_axis.erase(other_axis.begin() + std::min(seq_axis_, batch_axis_));
  for (int m = 0; m < other_count; ++m) {
    vector<int> other_indices = indices(m, other_shape);
    int other_idx = offset(bottom, other_axis, other_indices);
    for (int i = 0; i < batch_size; ++i) {
      int common_idx = other_idx + i * bottom[0]->count(batch_axis_ + 1);
      for (int j = 0; j < seq_lengths_[i]; ++j) {
        int b_idx = common_idx + j * bottom[0]->count(seq_axis_ + 1);
        int t_idx = common_idx +
                    (seq_lengths_[i] - 1 - j) * bottom[0]->count(seq_axis_ + 1);
        top_data[t_idx] = bottom_data[b_idx];
      }
    }
  }
}

INSTANTIATE_CLASS(ReverseSequenceLayer);
REGISTER_LAYER_CLASS(ReverseSequence);

} // namespace caffe
