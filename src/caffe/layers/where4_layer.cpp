#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/where4_layer.hpp"

namespace caffe {

template <typename Dtype>
void Where4Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Where4Parameter& where4_param = this->layer_param_.where4_param();
  num_output_ = where4_param.num_output();
  axis_ = bottom[0]->CanonicalAxisIndex(where4_param.axis());
  CHECK_GE(num_output_, 1) << "num_output must not be less than 1.";
  CHECK_GE(axis_, 0) << "axis must not be less than 0.";
  CHECK_LE(axis_, bottom[0]->num_axes()) <<
    "axis must be less than or equal to the number of axis.";
  CHECK_LE(num_output_, bottom[0]->shape(axis_))
    << "num_output must be less than or equal to the dimension of the axis.";
}

template <typename Dtype>
void Where4Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> shape(2);
  shape[0] = num_output_;
  shape[1] = bottom[0]->num_axes();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void Where4Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<int> group, group1, group2, group3;
  // TODO: add handling for other dimension conditions
  if (bottom[0]->num_axes() == 2)
  {
	if (axis_ == 1)
	{
	  for (int i=0; i<bottom[0]->shape(0); i++)
	  {
		for (int j=0; j<bottom[0]->shape(1);j++)
		{
			int value = int(bottom_data[i*bottom[0]->shape(1)+j]);
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
		  top_data[i*bottom[0]->shape(1)+j*2] = i;
		  top_data[i*bottom[0]->shape(1)+j*2+1] = group[j];
		}
		for (int j=0; j<group1.size() && (j+group.size())<num_output_; j++){
		  top_data[i*bottom[0]->shape(1)+(j+group.size())*2] = i;
		  top_data[i*bottom[0]->shape(1)+(j+group.size())*2+1] = group1[j];
		}
		for (int j=0; j<group2.size() && (j+group.size()+group1.size())<num_output_; j++){
		  top_data[i*bottom[0]->shape(1)+(j+group.size()+group1.size())*2] = i;
		  top_data[i*bottom[0]->shape(1)+(j+group.size()+group1.size())*2+1] = group2[j];
		}
		for (int j=0; j<group3.size() && (j+group.size()+group1.size()+group2.size())<num_output_; j++){
		  top_data[i*bottom[0]->shape(1)+(j+group.size()+group1.size()+group2.size())*2] = i;
		  top_data[i*bottom[0]->shape(1)+(j+group.size()+group1.size()+group2.size())*2+1] = group3[j];
		}
	  }
	}
  }
}

INSTANTIATE_CLASS(Where4Layer);
REGISTER_LAYER_CLASS(Where4);

}  // namespace caffe
