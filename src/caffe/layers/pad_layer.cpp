#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/pad_layer.hpp"

namespace caffe {

template <typename Dtype>
void PadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const PadParameter& pad_param = this->layer_param_.pad_param();
  constant_values_ = pad_param.constant_values();
  paddings_.clear();
  std::copy(pad_param.paddings().begin(),
	  pad_param.paddings().end(),
	  std::back_inserter(paddings_));
  int pad_dim = paddings_.size();
  CHECK_EQ(pad_dim % 2, 0) << "Paddings for each dimension should have 2 values!";
  CHECK_EQ(pad_dim / 2, bottom[0]->num_axes()) << "Paddings' num should be 2 times of bottom dimension!";
  CHECK_LE(bottom[0]->num_axes(), 4) << "Not support more than 4D paddings!";
}

template <typename Dtype>
void PadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_top_axes = bottom[0]->num_axes();
  std::vector<int> shape(num_top_axes, 1);
  shape = bottom[0]->shape();
  for (int i=0;i<num_top_axes;i++)
  {
	shape[i] = shape[i] + paddings_[2*i] + paddings_[2*i+1];
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(constant_values_), top_data);
  const int num_top_axes = bottom[0]->num_axes();
  if (num_top_axes == 4)
  {
    for (int i=0; i<bottom[0]->shape(0); i++)
    {
      for (int j=0; j<bottom[0]->shape(1); j++)
      {
    	for (int k=0; k<bottom[0]->shape(2); k++)
    	{
    	  for (int l=0; l<bottom[0]->shape(3); l++)
    	  {
    		int bottom_index = ((i * bottom[0]->shape(1) + j) * bottom[0]->shape(2) + k) * bottom[0]->shape(3) + l;
    		int top_index = (((i + paddings_[0]) * (paddings_[2] + paddings_[3] + bottom[0]->shape(1)) +
							   j + paddings_[2]) * (paddings_[4] + paddings_[5] + bottom[0]->shape(2)) +
							   k + paddings_[4]) * (paddings_[6] + paddings_[7] + bottom[0]->shape(3)) +
    						   l + paddings_[6];
    		top_data[top_index] = bottom_data[bottom_index];
    	  }
    	}
      }
    }
  }
  if (num_top_axes == 3)
  {
    for (int i=0; i<bottom[0]->shape(0); i++)
    {
      for (int j=0; j<bottom[0]->shape(1); j++)
      {
    	for (int k=0; k<bottom[0]->shape(2); k++)
    	{
    	  int bottom_index = (i * bottom[0]->shape(1) + j) * bottom[0]->shape(2) + k;
    	  int top_index = ((i + paddings_[0]) * (paddings_[2] + paddings_[3] +  bottom[0]->shape(1)) +
    						j + paddings_[2]) * (paddings_[4] + paddings_[5] + bottom[0]->shape(2)) +
    						k + paddings_[4];
    	  top_data[top_index] = bottom_data[bottom_index];
    	}
      }
    }
  }
  if (num_top_axes == 2)
  {
    for (int i=0; i<bottom[0]->shape(0); i++)
    {
      for (int j=0; j<bottom[0]->shape(1); j++)
      {
    	int bottom_index = i * bottom[0]->shape(1) + j;
    	int top_index = (i + paddings_[0]) * (paddings_[2] + paddings_[3] + bottom[0]->shape(1)) +
    					 j + paddings_[2];
    	top_data[top_index] = bottom_data[bottom_index];
      }
    }
  }
  if (num_top_axes == 1)
  {
    for (int i=0; i<bottom[0]->shape(0); i++)
    {
    	int bottom_index = i;
    	int top_index = i + paddings_[0];
    	top_data[top_index] = bottom_data[bottom_index];
    }
  }
}

INSTANTIATE_CLASS(PadLayer);
REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe
