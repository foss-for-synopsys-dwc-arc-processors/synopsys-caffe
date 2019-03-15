#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/topk_gather_layer.hpp"

namespace caffe {

template <typename Dtype>
void TopkGatherLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const TopkGatherParameter& topk_gather_param = this->layer_param_.topk_gather_param();
  top_k_ = topk_gather_param.top_k();
  has_axis_ = topk_gather_param.has_axis();
  CHECK_GE(top_k_, 1) << "top k must not be less than 1.";
  if (has_axis_) {
    axis_ = bottom[0]->CanonicalAxisIndex(topk_gather_param.axis());
    CHECK_GE(axis_, 0) << "axis must not be less than 0.";
    CHECK_LE(axis_, bottom[0]->num_axes()) <<
      "axis must be less than or equal to the number of axis.";
    CHECK_LE(top_k_, bottom[0]->shape(axis_))
      << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    CHECK_LE(top_k_, bottom[0]->count(1))
      << "top_k must be less than or equal to"
        " the dimension of the flattened bottom blob per instance.";
  }
}

template <typename Dtype>
void TopkGatherLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_top_axes = bottom[0]->num_axes();
  if ( num_top_axes < 3 ) num_top_axes = 3;
  std::vector<int> shape(num_top_axes, 1);
  if (has_axis_) {
    // Produces max_ind or max_val per axis
    shape = bottom[0]->shape();
    shape[axis_] = top_k_;
  } else {
    shape[0] = bottom[0]->shape(0);
    // Produces max_ind
    shape[2] = top_k_;
  }
  topk_indices_.Reshape(shape);

  const int num_axes = bottom[0]->num_axes();
  CHECK_GE(num_axes, 1) << "the dimension of input should be larger than or equal to 1";
  const TopkGatherParameter& topk_gather_param = this->layer_param_.topk_gather_param();
  gather_axis_ = bottom[0]->CanonicalAxisIndex(topk_gather_param.axis());
  indices_shape_ = topk_indices_.shape();

  if (indices_shape_.size() == 1 && indices_shape_[0] == 0) {
 	indices_dim_ = 0;
  }
  else {
	indices_dim_ = indices_shape_.size();
	int count = 1;
	for (int i = 0; i < indices_shape_.size(); ++i) {
	  count *= indices_shape_[i];
	}
  }
  // Initialize with the first blob
  // The result shape is params.shape[-1:axis] + indices.shape +
  // params.shape[axis + 0:].
  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(bottom_shape.size() + indices_dim_ - 1);
  num_gather_ = bottom[0]->count(0, gather_axis_);
  gather_size_ = bottom[0]->count(gather_axis_ + 1);

  for (int i = 0; i < gather_axis_; ++i) {
	top_shape[i] = bottom_shape[i];
  }
  for (int i = 0; i < indices_dim_; ++i) {
	top_shape[i + gather_axis_] = indices_shape_[i];
  }
  for (int i = gather_axis_ + 1; i < num_axes; ++i) {
	top_shape[i + indices_dim_ - 1] = bottom_shape[i];
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void TopkGatherLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* topk_indices_data = topk_indices_.mutable_cpu_data();
  int dim, axis_dist;
  if (has_axis_) {
    dim = bottom[0]->shape(axis_);
    // Distance between values of axis in blob
    axis_dist = bottom[0]->count(axis_) / dim;
  } else {
    dim = bottom[0]->count(1);
    axis_dist = 1;
  }
  int num = bottom[0]->count() / dim;
  std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector[j] = std::make_pair(
        bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    for (int j = 0; j < top_k_; ++j) {
        // Produces max_ind per axis
    	topk_indices_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
          = bottom_data_vector[j].second;
    }
  }

  vector<int> bottom_shape = bottom[0]->shape();
  const Dtype* bottom_data_c = bottom[0]->cpu_data(); //reset pointer
  Dtype* top_data = top[0]->mutable_cpu_data();
  topk_indices_data = topk_indices_.mutable_cpu_data();
  const int bottom_gather_axis = bottom[0]->shape(gather_axis_);
  int num_c = 0;
  for (int m = 0; m < num_gather_; ++m) {
	for (int n = 0; n < top_k_; ++n) {
	  const int top_offset = num_c * gather_size_;
      const int bottom_offset =
		  (m * bottom_gather_axis + topk_indices_data[n]) * gather_size_;
      caffe_copy(gather_size_,
		  bottom_data_c + bottom_offset, top_data + top_offset);
      num_c += 1;
	}
  }
}

INSTANTIATE_CLASS(TopkGatherLayer);
REGISTER_LAYER_CLASS(TopkGather);

}  // namespace caffe
