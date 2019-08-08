#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/embedding_lookup_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EmbeddingLookupLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const EmbeddingLookupParameter &embedding_lookup_param =
      this->layer_param_.embedding_lookup_param();
  ids.clear();
  std::copy(embedding_lookup_param.ids().begin(),
            embedding_lookup_param.ids().end(), std::back_inserter(ids));
  ids_shape.clear();
  std::copy(embedding_lookup_param.ids_shape().begin(),
            embedding_lookup_param.ids_shape().end(),
            std::back_inserter(ids_shape));
  p_strategy = embedding_lookup_param.partition_strategy();

  // if max_norm != None
  if (embedding_lookup_param.has_max_norm()) {
    max_norm = embedding_lookup_param.max_norm();
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
}

template <typename Dtype>
void EmbeddingLookupLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape.erase(top_shape.begin());
  top_shape.insert(top_shape.begin(), ids_shape.begin(), ids_shape.end());
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EmbeddingLookupLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const EmbeddingLookupParameter &embedding_lookup_param =
      this->layer_param_.embedding_lookup_param();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int copy_num = bottom[0]->count(1);
  // for one params
  if (bottom.size() == 1) {
    const Dtype *bottom_data = bottom[0]->cpu_data();
    for (int i = 0; i < ids.size(); ++i) {
      CHECK_GE(ids[i], 0) << "ids[" << i << "] = " << ids[i] << " is not in "
                          << "[0, " << bottom[0]->shape(0) << ").";
      CHECK_LT(ids[i], bottom[0]->shape(0))
          << "ids[" << i << "] = " << ids[i] << " is not in "
          << "[0, " << bottom[0]->shape(0) << ").";
      const int t_offset = i * copy_num;
      const int b_offset = ids[i] * copy_num;
      caffe_copy(copy_num, bottom_data + b_offset, top_data + t_offset);
      const auto normt = std::sqrt(caffe_cpu_dot(
          copy_num, bottom_data + b_offset, bottom_data + b_offset));
      if ((embedding_lookup_param.has_max_norm()) && (normt > max_norm)) {
        const auto alpha = max_norm / normt;
        caffe_scal(copy_num, alpha, top_data + t_offset);
      }
    }
  }
  // for multiple params
  else {
    for (int i = 0; i < ids.size(); ++i) {
      // strategy = mod
      if (p_strategy == "mod") {
        const int bottom_num = ids[i] % bottom.size();
        const int row_num = ids[i] / bottom.size();
        CHECK_GE(row_num, 0) << "ids[" << i << "] is not in "
                             << "[0, " << bottom[bottom_num]->shape(0)
                             << ") for params[" << bottom_num << "]";
        CHECK_LT(row_num, bottom[bottom_num]->shape(0))
            << "ids[" << i << "] is not in "
            << "[0, " << bottom[bottom_num]->shape(0) << ") for params["
            << bottom_num << "]";
        const Dtype *bottom_data = bottom[bottom_num]->cpu_data();
        const int t_offset = i * copy_num;
        const int b_offset = row_num * copy_num;
        caffe_copy(copy_num, bottom_data + b_offset, top_data + t_offset);
        // max_norm part
        const auto normt = std::sqrt(caffe_cpu_dot(
            copy_num, bottom_data + b_offset, bottom_data + b_offset));
        if ((embedding_lookup_param.has_max_norm()) && (normt > max_norm)) {
          const auto alpha = max_norm / normt;
          caffe_scal(copy_num, alpha, top_data + t_offset);
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
        const int bottom_num = (ids[i] < b * (a + 1))
                                   ? (ids[i] / (a + 1))
                                   : (b + (ids[i] - b * (a + 1)) / a);
        const int row_num = (ids[i] < b * (a + 1))
                                ? (ids[i] % (a + 1))
                                : ((ids[i] - b * (a + 1)) % a);
        CHECK_GE(row_num, 0) << "ids[" << i << "] is not in "
                             << "[0, " << bottom[bottom_num]->shape(0)
                             << ") for params[" << bottom_num << "]";
        CHECK_LT(row_num, bottom[bottom_num]->shape(0))
            << "ids[" << i << "] is not in "
            << "[0, " << bottom[bottom_num]->shape(0) << ") for params["
            << bottom_num << "]";
        const Dtype *bottom_data = bottom[bottom_num]->cpu_data();
        const int t_offset = i * copy_num;
        const int b_offset = row_num * copy_num;
        caffe_copy(copy_num, bottom_data + b_offset, top_data + t_offset);
        // max_norm part
        const auto normt = std::sqrt(caffe_cpu_dot(
            copy_num, bottom_data + b_offset, bottom_data + b_offset));
        if ((embedding_lookup_param.has_max_norm()) && (normt > max_norm)) {
          const auto alpha = max_norm / normt;
          caffe_scal(copy_num, alpha, top_data + t_offset);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(EmbeddingLookupLayer);
REGISTER_LAYER_CLASS(EmbeddingLookup);

} // namespace caffe
