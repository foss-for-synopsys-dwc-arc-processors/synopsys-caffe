#include <algorithm>
#include <vector>

#include "caffe/layers/piece_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PieceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  const PieceParameter &piece_param = this->layer_param_.piece_param();

  piece_begin_.clear();
  std::copy(piece_param.begin().begin(), piece_param.begin().end(),
            std::back_inserter(piece_begin_));

  piece_size_.clear();
  std::copy(piece_param.size().begin(), piece_param.size().end(),
            std::back_inserter(piece_size_));

  CHECK_EQ(piece_begin_.size(), bottom[0]->num_axes())
      << "the length of begin and input should be the same!";
  CHECK_EQ(piece_begin_.size(), piece_size_.size())
      << "the length of begin and size should be the same!";

  for (int i = 0; i < piece_begin_.size(); ++i) {
    if (piece_size_[i] == -1) {
      piece_size_[i] = bottom[0]->shape(i) - piece_begin_[i];
    }
    CHECK_GE(piece_begin_[i], 0)
        << "the element in begin should not be negative!";
    CHECK_GE(piece_size_[i], 1) << "the element in size should be positive!";
    CHECK_LE(piece_begin_[i] + piece_size_[i], bottom[0]->shape(i))
        << "the sum of begin and size should not be greater than input shape";
  }
}

template <typename Dtype>
void PieceLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {

  vector<int> top_shape = piece_size_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline vector<int> PieceLayer<Dtype>::piece(const vector<int> &begin,
                                            const vector<int> &size) const {
  std::vector<int> piece_idx;
  for (int i = 0; i < begin.size(); ++i) {
    for (int j = 0; j < size[i]; ++j) {
      piece_idx.insert(piece_idx.end(), begin[i] + j);
    }
  }
  return piece_idx;
}

template <typename Dtype>
inline vector<int>
PieceLayer<Dtype>::Indices(int offset, const vector<int> &top_shape,
                           const vector<int> &piece_idx) const {
  vector<int> indices0(top_shape.size(), 0);
  vector<int> indices1(top_shape.size(), 0);
  int r = offset;
  int ts = 0;
  for (int i = top_shape.size() - 1; i >= 0; i--) {
    indices0[i] = r % top_shape[i];
    r /= top_shape[i];
  }
  for (int i = 0; i < top_shape.size(); ++i) {
    indices1[i] = piece_idx[ts + indices0[i]];
    ts = ts + top_shape[i];
  }
  return indices1;
}

template <typename Dtype>
inline int PieceLayer<Dtype>::offset(const vector<Blob<Dtype> *> &bottom,
                                     const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < indices.size(); ++i) {
    offset += indices[i] * bottom[0]->count(i + 1);
  }
  return offset;
}

template <typename Dtype>
void PieceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  vector<int> top_shape = top[0]->shape();

  std::vector<int> piece_idx = piece(piece_begin_, piece_size_);

  for (int i = 0; i < top[0]->count(); ++i) {
    vector<int> indices = Indices(i, top_shape, piece_idx);
    int b_idx = offset(bottom, indices);
    top_data[i] = bottom_data[b_idx];
  }
}

INSTANTIATE_CLASS(PieceLayer);
REGISTER_LAYER_CLASS(Piece);

} // namespace caffe
