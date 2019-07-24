#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/sparse_to_dense_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SparseToDenseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const SparseToDenseParameter &sparse_to_dense_param =
      this->layer_param_.sparse_to_dense_param();
  sparse_indices_.clear();
  std::copy(sparse_to_dense_param.sparse_indices().begin(),
            sparse_to_dense_param.sparse_indices().end(),
            std::back_inserter(sparse_indices_));
  sparse_indices_shape_.clear();
  std::copy(sparse_to_dense_param.sparse_indices_shape().begin(),
            sparse_to_dense_param.sparse_indices_shape().end(),
            std::back_inserter(sparse_indices_shape_));
  output_shape_.clear();
  std::copy(sparse_to_dense_param.output_shape().begin(),
            sparse_to_dense_param.output_shape().end(),
            std::back_inserter(output_shape_));
  sparse_values_.clear();
  std::copy(sparse_to_dense_param.sparse_values().begin(),
            sparse_to_dense_param.sparse_values().end(),
            std::back_inserter(sparse_values_));
  default_value_ = (sparse_to_dense_param.has_default_value())
                       ? sparse_to_dense_param.default_value()
                       : 0;
  validate_indices_ = (sparse_to_dense_param.has_validate_indices())
                          ? sparse_to_dense_param.validate_indices()
                          : true;

  int shape_count = 1;
  for (int i = 0; i < sparse_indices_shape_.size(); ++i) {
    shape_count *= sparse_indices_shape_[i];
  }
  CHECK_LE(sparse_indices_shape_.size(), 2)
      << "sparse_indices should be a scalar, vector, or matrix";
  CHECK_EQ(sparse_indices_.size(), shape_count)
      << "sparse_indices and sparse_indices_shape do not match!";
  if (sparse_indices_shape_[0] != 0) {
    if (sparse_values_.size() == 1) {
      const int value = sparse_values_[0];
      sparse_values_.assign(sparse_indices_shape_[0], value);
    }
    CHECK_EQ(sparse_indices_shape_[0], sparse_values_.size())
        << "sparse_values has incorrect shape: " << sparse_values_.size()
        << ", should be: " << sparse_indices_shape_[0];

    if (sparse_indices_shape_.size() == 1) {
      CHECK_EQ(output_shape_.size(), 1) << "output_shape should be 1D tensor";
      vector<int>::iterator max =
          std::max_element(sparse_indices_.begin(), sparse_indices_.end());
      CHECK_LT(*max, output_shape_[0]) << "sparse_indices is out of bounds";
    } else {
      CHECK_EQ(sparse_indices_shape_[1], output_shape_.size())
          << "output_shape has incorrect number of elements: "
          << output_shape_.size()
          << ", should be: " << sparse_indices_shape_[1];
    }
  } else {
    CHECK_EQ(sparse_values_.size(), 1)
        << "sparse_values should have only one element";
    CHECK_EQ(output_shape_.size(), 1) << "output should be 1-D Tensor";
    CHECK_LT(sparse_indices_[0], output_shape_[0])
        << "sparse_indices is out of bounds";
  }

  if (validate_indices_ && sparse_indices_shape_[0] != 0) {
    if (sparse_indices_shape_.size() == 1) {
      for (int i = 0; i < sparse_indices_.size() - 1; ++i) {
        CHECK_LT(sparse_indices_[i], sparse_indices_[i + 1])
            << "sparse_indices[" << i + 1 << "] is out of order or repeated";
      }
    } else {
      const int interval = sparse_indices_shape_[1];
      for (int i = 0; i < sparse_indices_.size() - interval; i = i + interval) {
        CHECK_LT(sparse_indices_[i], sparse_indices_[i + interval])
            << "sparse_indices is out of order or repeated";
      }
    }
  }
}

template <typename Dtype>
void SparseToDenseLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  top[0]->Reshape(output_shape_);
}

template <typename Dtype>
inline int SparseToDenseLayer<Dtype>::count(const vector<int> &shape,
                                            const int axis) const {
  int count = 1;
  for (int i = axis; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
}

template <typename Dtype>
void SparseToDenseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(default_value_), top_data);
  // sparse_indices is scalar or vector
  if (sparse_indices_shape_.size() == 1) {
    for (int i = 0; i < sparse_indices_.size(); ++i) {
      top_data[sparse_indices_[i]] = sparse_values_[i];
    }
  }
  // sparse_indices is matrix
  if (sparse_indices_shape_.size() == 2) {
    const int m = sparse_indices_shape_[0];
    const int n = sparse_indices_shape_[1];
    for (int i = 0; i < m; ++i) {
      int t_idx = 0;
      for (int j = 0; j < n; ++j) {
        int offset = i * n + j;
        CHECK_LT(sparse_indices_[offset], output_shape_[j])
            << "sparse_indices is out of bounds";
        t_idx += sparse_indices_[offset] * count(output_shape_, j + 1);
      }
      top_data[t_idx] = sparse_values_[i];
    }
  }
}

INSTANTIATE_CLASS(SparseToDenseLayer);
REGISTER_LAYER_CLASS(SparseToDense);

} // namespace caffe
