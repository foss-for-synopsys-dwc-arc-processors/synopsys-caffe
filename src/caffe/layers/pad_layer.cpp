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
  //CHECK_LE(bottom[0]->num_axes(), 4) << "Not support more than 4D paddings!";
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

// Recursive program to support padding of arbitrary number of axes.
// Initially suggested by Tom Pennello
template <typename Dtype>
void PadLayer<Dtype>::Pad(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int level, int bottom_index, int top_index, vector<int> paddings) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	vector<int> BS = bottom[0]->shape();
	vector<int> TS = top[0]->shape();

	bool innermost = (level == BS.size()-1);
    int bottom_level_size = BS[level];
    //int top_level_size = TS[level];
    int padl = paddings[level*2];

    for (int ix = 0; ix < bottom_level_size; ix++) {
        int bix = bottom_index + ix;
        int tix = top_index + padl + ix;
        if (innermost) {
            //printf("top_data[%d] <- bottom_data[%d]\n",tix,bix); //Show for debug
            top_data[tix] = bottom_data[bix];
        }
        else {
            Pad(bottom, top, level+1, bix*BS[level+1], tix*TS[level+1], paddings);
        }
    }
};

template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(constant_values_), top_data);
  //const int num_top_axes = bottom[0]->num_axes();
  //vector<int> BS = bottom[0]->shape();
  //vector<int> TS = top[0]->shape();

  Pad(bottom, top, 0,0,0, paddings_);

  /*
  if (num_top_axes == 4)
  {
    int P0 = paddings_[0];
    int P2 = paddings_[2];
    int P4 = paddings_[4];
    int P6 = paddings_[6];

    for (int i=0; i<BS[0]; i++)
    {
      for (int j=0; j<BS[1]; j++)
      {
    	for (int k=0; k<BS[2]; k++)
    	{
    	  for (int l=0; l<BS[3]; l++)
    	  {
    		int bottom_index = ((i * BS[1] + j) * BS[2] + k) * BS[3] + l;
    		int top_index = (((i + P0) * TS[1] + j + P2) * TS[2] + k + P4) * TS[3] + l + P6;
    		top_data[top_index] = bottom_data[bottom_index];
    	  }
    	}
      }
    }
  }
  if (num_top_axes == 3)
  {
    int P0 = paddings_[0];
    int P2 = paddings_[2];
    int P4 = paddings_[4];
    for (int i=0; i<BS[0]; i++)
    {
      for (int j=0; j<BS[1]; j++)
      {
    	for (int k=0; k<BS[2]; k++)
    	{
    	  int bottom_index = (i * BS[1] + j) * BS[2] + k;
    	  int top_index = ((i + P0) * TS[1] + j + P2) * TS[2] + k + P4;
    	  top_data[top_index] = bottom_data[bottom_index];
    	}
      }
    }
  }
  if (num_top_axes == 2)
  {
    int P0 = paddings_[0];
    int P2 = paddings_[2];
    for (int i=0; i<BS[0]; i++)
    {
      for (int j=0; j<BS[1]; j++)
      {
    	int bottom_index = i * BS[1] + j;
    	int top_index = (i + P0) * TS[1] + j + P2;
    	top_data[top_index] = bottom_data[bottom_index];
      }
    }
  }
  if (num_top_axes == 1)
  {
	int P0 = paddings_[0];
    for (int i=0; i<bottom[0]->shape(0); i++)
    {
    	int bottom_index = i;
    	int top_index = i + P0;
    	top_data[top_index] = bottom_data[bottom_index];
    }
  }
  */
}

INSTANTIATE_CLASS(PadLayer);
REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe
