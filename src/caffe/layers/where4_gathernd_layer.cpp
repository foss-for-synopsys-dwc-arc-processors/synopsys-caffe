#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/layers/where4_gathernd_layer.hpp"
#include "caffe/util/math_functions.hpp"      



namespace caffe {

template <typename Dtype>
void Where4GatherndLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
	 const Where4GatherndParameter& where4_gathernd_param = this->layer_param_.where4_gathernd_param();
  num_output_ = where4_gathernd_param.num_output();
  axis_ = bottom[1]->CanonicalAxisIndex(where4_gathernd_param.axis());
  CHECK_GE(num_output_, 1) << "num_output must not be less than 1.";
  CHECK_GE(axis_, 0) << "axis must not be less than 0.";
  CHECK_LE(axis_, bottom[1]->num_axes()) <<
	"axis must be less than or equal to the number of axis.";
  CHECK_LE(num_output_, bottom[1]->shape(axis_))
	<< "num_output must be less than or equal to the dimension of the axis.";
}

template <typename Dtype>
void Where4GatherndLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
	std::vector<int> shape(2);
	shape[0] = num_output_;
	shape[1] = bottom[1]->num_axes();
	indices_shape_.clear();
	for(int i=0;i<shape.size();i++)
	  indices_shape_.push_back(shape[i]);

 	const int num_axes = bottom[0]->num_axes();
 	CHECK_GE(num_axes, 1) << "the dimension of input should be larger than or equal to 1";
 	indices_dim_ = indices_shape_.size();                          
 	CHECK_GE(indices_dim_, 1) << "the dimension of indices should be larger than or equal to 1";
 	int count = 1;
 	for (int i = 0; i < indices_shape_.size(); ++i) {
	  	count *= indices_shape_[i];
	 }
 	vector<int> bottom_shape = bottom[0]->shape();
 	vector<int> top_shape = bottom[0]->shape();
 	indices_N_ = indices_shape_[indices_shape_.size()-1];
 	CHECK_LE(indices_N_, num_axes) << "indices.shape[-1] must be <= params.rank, but saw indices.shape[-1]:" 
	    	<< indices_N_ << ", and params.rank: " << num_axes;
 	top_shape.resize(indices_dim_ - 1 + num_axes - indices_N_);     
 	gather_nd_size_ = bottom[0]->count(indices_N_);            

 	// The result shape is
 	//   indices.shape[:-1] + params.shape[indices.shape[-1]:]
	 for (int i = 0; i < indices_dim_ - 1; ++i) {
		  top_shape[i] = indices_shape_[i];
 	}
	 for (int i = 0; i < num_axes - indices_N_; ++i) {
		  top_shape[i + indices_dim_ - 1] = bottom_shape[i + indices_N_];
 	}
	 top[0]->Reshape(top_shape);
}

template <typename Dtype>
void Where4GatherndLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
  indices_.clear();
  const Dtype* indices_data = bottom[1]->cpu_data();
  vector<int> group, group1, group2, group3;
  // TODO: add handling for other dimension conditions
  if (bottom[1]->num_axes() == 2)
  {
	if (axis_ == 1)
	{
	  for (int i=0; i<bottom[1]->shape(0); i++)
	  {
		for (int j=0; j<bottom[1]->shape(1);j++)
		{
			int value = int(indices_data[i*bottom[1]->shape(1)+j]);
			switch(value){
			  case 2:
				group.push_back(j);
				break;
			  case 3:
				group1.push_back(j);
				break;
			  case 4:
				group2.push_back(j);
				break;
			  case 5:
				group3.push_back(j);
				break;
			  default:
				LOG(FATAL) << "The value "<<value<<" can't be sorted in any condition.";
				break;
			}
		}
		for (int j=0; j<group.size() && j<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group[j]);
		}
		for (int j=0; j<group1.size() && (j+group.size())<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group1[j]);
		}
		for (int j=0; j<group2.size() && (j+group.size()+group1.size())<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group2[j]);
		}
		for (int j=0; j<group3.size() && (j+group.size()+group1.size()+group2.size())<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group3[j]);
		}
	  }
	}
  }

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

INSTANTIATE_CLASS(Where4GatherndLayer);
REGISTER_LAYER_CLASS(Where4Gathernd);


}  // namespace caffe
