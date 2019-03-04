#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/layers/gather_layer.hpp"
#include "caffe/util/math_functions.hpp"      

namespace caffe {

template <typename Dtype>
void GatherLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  const GatherParameter& gather_param = this->layer_param_.gather_param();
  indices_.clear();
  std::copy(gather_param.indices().begin(),
	  gather_param.indices().end(),
	  std::back_inserter(indices_));
  indices_shape_.clear();
  std::copy(gather_param.shape().begin(),
	  gather_param.shape().end(),
	  std::back_inserter(indices_shape_));
}

template <typename Dtype>
void GatherLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  CHECK_GE(num_axes, 1) << "the dimension of input should be larger than or equal to 1";
  const GatherParameter& gather_param = this->layer_param_.gather_param();
  gather_axis_ = bottom[0]->CanonicalAxisIndex(gather_param.axis());
  if (indices_shape_.size() == 1 && indices_shape_[0] == 0) {         
 	indices_dim_ = 0;
	CHECK_EQ(indices_.size(), 1) << "indices should be scalar!";
  }
  else {
	indices_dim_ = indices_shape_.size();
	int count = 1;
	for (int i = 0; i < indices_shape_.size(); ++i) {
	  count *= indices_shape_[i];
	}
	CHECK_EQ(indices_.size(), count) << "the size and shape of indices do not match";
  }

  // Initialize with the first blob 
  // The result shape is params.shape[-1:axis] + indices.shape +
  // params.shape[axis + 0:].
  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(bottom_shape.size() + indices_dim_ - 1);     
  num_gather_ = bottom[0]->count(0, gather_axis_);
  gather_size_ = bottom[0]->count(gather_axis_ + 1);
  for (int i = 0; i < indices_.size(); ++i) {
	CHECK_GE(indices_[i], 0) << "indices_ element with idx" << i << " is negative";
	CHECK_LT(indices_[i], bottom[0]->shape(gather_axis_))
		<< "indices_ element with idx" << i << " is out of range "
		<< bottom[0]->shape(gather_axis_);
  }
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
void GatherLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  const Dtype* params = bottom[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int bottom_gather_axis = bottom[0]->shape(gather_axis_);
  int num = 0;
  for (int m = 0; m < num_gather_; ++m) {
	for (int n = 0; n < indices_.size(); ++n) {
	  const int top_offset = num * gather_size_;
      const int bottom_offset =
		  (m * bottom_gather_axis + indices_[n]) * gather_size_;
      caffe_copy(gather_size_,
		  bottom_data + bottom_offset, top_data + top_offset);
      num += 1;
	}
  }
}

INSTANTIATE_CLASS(GatherLayer);
REGISTER_LAYER_CLASS(Gather);

}  // namespace caffe
