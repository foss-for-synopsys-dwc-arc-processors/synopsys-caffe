#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/embedding_lookup_sparse_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EmbeddingLookupSparseLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const EmbeddingLookupSparseParameter &embedding_lookup_sparse_param =
      this->layer_param_.embedding_lookup_sparse_param();
  ids_id.clear();
  std::copy(embedding_lookup_sparse_param.ids_indices().begin(),
            embedding_lookup_sparse_param.ids_indices().end(),
            std::back_inserter(ids_id));
  ids_value.clear();
  std::copy(embedding_lookup_sparse_param.ids_values().begin(),
            embedding_lookup_sparse_param.ids_values().end(),
            std::back_inserter(ids_value));
  ids_shape.clear();
  std::copy(embedding_lookup_sparse_param.ids_shape().begin(),
            embedding_lookup_sparse_param.ids_shape().end(),
            std::back_inserter(ids_shape));
  w_value.clear();
  std::copy(embedding_lookup_sparse_param.weight_values().begin(),
            embedding_lookup_sparse_param.weight_values().end(),
            std::back_inserter(w_value));

  p_strategy = embedding_lookup_sparse_param.partition_strategy();
  combiner = embedding_lookup_sparse_param.combiner();
  max_norm = embedding_lookup_sparse_param
                 .max_norm(); // default value as None:9999999999

  if (combiner == "None") {
    combiner = "sqrtn";
  }

  // CHECK PART
  const int num_axes = bottom[0]->num_axes();
  vector<int> bottom_shape = bottom[0]->shape();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 1; j < num_axes; ++j) {
      CHECK_EQ(bottom_shape[j], bottom[i]->shape(j))
          << "Dimension " << j - 1 << " in both shapes must be equal, but are "
          << bottom_shape[j] << " and " << bottom[i]->shape(j);
    }
  }
  // ids_id, ids_value, ids_shape
  CHECK_EQ((ids_id.size() % 2), 0)
      << "ids_indices.size() should be even number";
  CHECK_EQ((ids_id.size() / 2), ids_value.size())
      << "sp_ids has incompatible indices and values";

  for (int i = 0; i < ids_id.size() - 2; i = i + 2) {
    CHECK_LE(ids_id[i], ids_id[i + 2]) << "segment ids are not increasing";
  }
  for (int i = 0; i < ids_id.size(); ++i) {
    CHECK_GE(ids_id[i], 0) << "segement ids must be >= 0";
    if (i % 2 == 0) {
      CHECK_LT(ids_id[i], ids_shape[0]) << "sp_ids's indices is out of bounds";
    } else {
      CHECK_LT(ids_id[i], ids_shape[1]) << "sp_ids's indices is out of bounds";
    }
  }
  // get weight_values, if None, set it all 1
  if (w_value[0] == 9999999999) {
    w_value.assign(ids_value.size(), 1);
  } else {
    CHECK_EQ(w_value.size(), ids_value.size())
        << "sp_ids and sp_weights are imcompatible ";
  }
}

template <typename Dtype>
void EmbeddingLookupSparseLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  const int len_ids = ids_id[ids_id.size() - 2] + 1;
  top_shape.erase(top_shape.begin());
  top_shape.insert(top_shape.begin(), len_ids);
  top[0]->Reshape(top_shape);
  top_pre.Reshape(ids_value.size(), bottom[0]->count(1), 1, 1);
}

template <typename Dtype>
void EmbeddingLookupSparseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Dtype *top_data = top[0]->mutable_cpu_data();
  Dtype *pre_data = top_pre.mutable_cpu_data();
  const int copy_num = bottom[0]->count(1);
  caffe_set(top[0]->count(), Dtype(0), top_data);
  caffe_set(top_pre.count(), Dtype(0), pre_data);
  // define none_value
  const float none_value = 9999999999;
  // calculate weight_sum and weight_suqared for combiner = mean/sqrtn
  vector<double> weight_sum(ids_id[ids_id.size() - 2] + 1, 0);
  vector<double> weight_squared(ids_id[ids_id.size() - 2] + 1, 0);
  int w_row = ids_id[0];
  for (int i = 0; i < w_value.size(); ++i) {
    if (w_row != ids_id[i * 2]) {
      w_row = ids_id[2 * i];
    }
    weight_sum[w_row] += w_value[i];
    weight_squared[w_row] += std::pow(w_value[i], 2);
  }

  int row = ids_id[0];
  // for one params
  if (bottom.size() == 1) {
    const Dtype *bottom_data = bottom[0]->cpu_data();
    for (int i = 0; i < ids_value.size(); ++i) {
      CHECK_GE(ids_value[i], 0)
          << "sp_ids.values[" << i << "] = " << ids_value[i] << " is not in "
          << "[0, " << bottom[0]->shape(0) << ").";
      CHECK_LT(ids_value[i], bottom[0]->shape(0))
          << "sp_ids.values[" << i << "] = " << ids_value[i] << " is not in "
          << "[0, " << bottom[0]->shape(0) << ").";
      const int p_offset = i * copy_num;
      const int b_offset = ids_value[i] * copy_num;
      // put data (selected by ids_value, normalized by max_norm) into pre_data
      caffe_copy(copy_num, bottom_data + b_offset, pre_data + p_offset);
      const auto normt = std::sqrt(caffe_cpu_dot(
          copy_num, bottom_data + b_offset, bottom_data + b_offset));
      if (max_norm != none_value && (normt > max_norm)) {
        caffe_scal(copy_num, Dtype(max_norm / normt), pre_data + p_offset);
      }
      // if row does not change, add them to top_data[row,:]
      if (row != ids_id[i * 2]) {
        row = ids_id[i * 2];
      }
      if (combiner == "sum") {
        caffe_axpy(copy_num, Dtype(w_value[i]), pre_data + p_offset,
                   top_data + row * copy_num);
      }
      if (combiner == "mean") {
        caffe_axpy(copy_num, Dtype(w_value[i] / weight_sum[row]),
                   pre_data + p_offset, top_data + row * copy_num);
      }
      if (combiner == "sqrtn") {
        caffe_axpy(copy_num, Dtype(w_value[i] / std::sqrt(weight_squared[row])),
                   pre_data + p_offset, top_data + row * copy_num);
      }
    }
  }
  // for multiple params
  else {
    for (int i = 0; i < ids_value.size(); ++i) {
      const int p_offset = i * copy_num;
      // strategy = mod
      if (p_strategy == "mod") {
        const int bottom_num = ids_value[i] % bottom.size();
        const int row_num = ids_value[i] / bottom.size();
        CHECK_GE(row_num, 0) << "ids_value[" << i << "] is not in "
                             << "[0, " << bottom[bottom_num]->shape(0)
                             << ") for params[" << bottom_num << "]";
        CHECK_LT(row_num, bottom[bottom_num]->shape(0))
            << "ids_value[" << i << "] is not in "
            << "[0, " << bottom[bottom_num]->shape(0) << ") for params["
            << bottom_num << "]";
        const Dtype *bottom_data = bottom[bottom_num]->cpu_data();
        const int b_offset = row_num * copy_num;
        caffe_copy(copy_num, bottom_data + b_offset, pre_data + p_offset);
        // max_norm part
        const auto normt = std::sqrt(caffe_cpu_dot(
            copy_num, bottom_data + b_offset, bottom_data + b_offset));
        if (max_norm != none_value && (normt > max_norm)) {
          caffe_scal(copy_num, Dtype(max_norm / normt), pre_data + p_offset);
        }
      }
      // strategy = div
      if (p_strategy == "div") {
        int all_idx = 0;
        for (int i = 0; i < bottom.size(); ++i) {
          all_idx += bottom[i]->shape(0);
        }
        const int a = all_idx / bottom.size();
        const int b = all_idx % bottom.size();
        const int bottom_num = (ids_value[i] < b * (a + 1))
                                   ? (ids_value[i] / (a + 1))
                                   : (b + (ids_value[i] - b * (a + 1)) / a);
        const int row_num = (ids_value[i] < b * (a + 1))
                                ? (ids_value[i] % (a + 1))
                                : ((ids_value[i] - b * (a + 1)) % a);
        CHECK_GE(row_num, 0) << "ids_value[" << i << "] is not in "
                             << "[0, " << bottom[bottom_num]->shape(0)
                             << ") for params[" << bottom_num << "]";
        CHECK_LT(row_num, bottom[bottom_num]->shape(0))
            << "ids_value[" << i << "] is not in "
            << "[0, " << bottom[bottom_num]->shape(0) << ") for params["
            << bottom_num << "]";
        const Dtype *bottom_data = bottom[bottom_num]->cpu_data();
        const int b_offset = row_num * copy_num;
        caffe_copy(copy_num, bottom_data + b_offset, pre_data + p_offset);
        // max_norm part
        const auto normt = std::sqrt(caffe_cpu_dot(
            copy_num, bottom_data + b_offset, bottom_data + b_offset));
        if (max_norm != none_value && (normt > max_norm)) {
          caffe_scal(copy_num, Dtype(max_norm / normt), pre_data + p_offset);
        }
      }

      // combiner part
      if (row != ids_id[i * 2]) {
        row = ids_id[i * 2];
      }
      if (combiner == "sum") {
        caffe_axpy(copy_num, Dtype(w_value[i]), pre_data + p_offset,
                   top_data + row * copy_num);
      }
      if (combiner == "mean") {
        caffe_axpy(copy_num, Dtype(w_value[i] / weight_sum[row]),
                   pre_data + p_offset, top_data + row * copy_num);
      }
      if (combiner == "sqrtn") {
        caffe_axpy(copy_num, Dtype(w_value[i] / std::sqrt(weight_squared[row])),
                   pre_data + p_offset, top_data + row * copy_num);
      }
    }
  }
}

INSTANTIATE_CLASS(EmbeddingLookupSparseLayer);
REGISTER_LAYER_CLASS(EmbeddingLookupSparse);

} // namespace caffe
