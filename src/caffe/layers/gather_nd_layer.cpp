#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/layers/gather_nd_layer.hpp"
#include "caffe/util/math_functions.hpp"      



namespace caffe {

template <typename Dtype>
void GatherNdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
	 const GatherNdParameter& gather_nd_param = this->layer_param_.gather_nd_param();
	 indices_.clear();
	 std::copy(gather_nd_param.indices().begin(),
		    gather_nd_param.indices().end(),
		    std::back_inserter(indices_));
	 indices_shape_.clear();
	 std::copy(gather_nd_param.shape().begin(),
		    gather_nd_param.shape().end(),
		    std::back_inserter(indices_shape_));
}

template <typename Dtype>
void GatherNdLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
 	const int num_axes = bottom[0]->num_axes();
 	CHECK_GE(num_axes, 1) << "the dimension of input should be larger than or equal to 1";
	//const GatherNdParameter& gather_nd_param = this->layer_param_.gather_nd_param();
 	indices_dim_ = indices_shape_.size();                          
 	CHECK_GE(indices_dim_, 1) << "the dimension of indices should be larger than or equal to 1";
 	int count = 1;
 	for (int i = 0; i < indices_shape_.size(); ++i) {
	  	count *= indices_shape_[i];
	 }
	 CHECK_EQ(indices_.size(), count) << "the size and shape of indices do not match" ;
 	vector<int> bottom_shape = bottom[0]->shape();
 	vector<int> top_shape = bottom[0]->shape();
 	indices_N_ = indices_shape_[indices_shape_.size()-1];
 	CHECK_LE(indices_N_, num_axes) << "indices.shape[-1] must be <= params.rank, but saw indices.shape[-1]:" 
	    	<< indices_N_ << ", and params.rank: " << num_axes;
 	top_shape.resize(indices_dim_ - 1 + num_axes - indices_N_);     
 	gather_nd_size_ = bottom[0]->count(indices_N_);            

 	// The result shape is
 	//   indices.shape[:-1] + params.shape[indices.shape[-1]:]
 	for (int i = 0; i < indices_.size(); ++i) {
	  	CHECK_GE(indices_[i], 0) << "indices_ element with idx" << i << " is negative";
 	}
	 for (int i = 0; i < indices_dim_ - 1; ++i) {
		  top_shape[i] = indices_shape_[i];
 	}
	 for (int i = 0; i < num_axes - indices_N_; ++i) {
		  top_shape[i + indices_dim_ - 1] = bottom_shape[i + indices_N_];
 	}
	 top[0]->Reshape(top_shape);
}

template <typename Dtype>
void GatherNdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
 	const Dtype* bottom_data = bottom[0]->cpu_data();
 	Dtype* top_data = top[0]->mutable_cpu_data();
 	vector<int> bottom_shape = bottom[0]->shape();
 	for (int m = 0; m < indices_.size()/indices_N_; ++m) {
	  	const int top_offset = m * gather_nd_size_;
	  	int bottom_offset = 0;
	  	for (int n = 0; n < indices_N_; ++n) {
		   	int indices_value = indices_[m*indices_N_ + n];
			   int params_idx = bottom_shape[n];
			   CHECK_LT(indices_value, params_idx) << "indices value does not index into param dimension: " << n;
			   bottom_offset += indices_[m*indices_N_ + n] * bottom[0]->count(n + 1);
		  } 
		  caffe_copy(gather_nd_size_,
		 	    bottom_data + bottom_offset, top_data + top_offset);
	 } 
}

INSTANTIATE_CLASS(GatherNdLayer);
REGISTER_LAYER_CLASS(GatherNd);


}  // namespace caffe
