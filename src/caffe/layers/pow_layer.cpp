#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/pow_layer.hpp"

namespace caffe {
template <typename Dtype>
void PowLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Add the fake initialization to avoid special case in evconvert for the moment when
  // the prototxt is generated, the 2nd input is parameter to be fed into caffemodel, but the caffemodel is not generated yet
  if (bottom.size() == 1 && this->blobs_.size() == 0)
  {
    LOG(WARNING) << "Note: Make fake parameter initialization to avoid segment fault!";
    // initialize the 2nd input to avoid error in reshape stage
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin();
    const vector<int>::const_iterator& shape_end = bottom[0]->shape().end();
    vector<int> second_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(second_shape));
    // Note: seems not to support broadcasting data value initialize when
    // input1 shape < input2 shape in this special situation

    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }
}

template <typename Dtype>
void PowLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not allow in-place computation.";
  is_scalar_=false;
  Blob<Dtype>* bottom1 = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();

  if(bottom[0]->num_axes()==0 || bottom1->num_axes()==0)
    is_scalar_=true;
  dim_diff_ = bottom[0]->num_axes() - bottom1->num_axes();
  dim_ = bottom[0]->num_axes() >= bottom1->num_axes() ? bottom[0]->num_axes() : bottom1->num_axes();
  vector<int> top_shape(dim_, 1);
  if(dim_diff_ == 0)
  {
    if(!is_scalar_)
    {
      for(int i=0;i<dim_;i++)
      {
        CHECK(bottom[0]->shape(i)==bottom1->shape(i) || bottom[0]->shape(i)==1 || bottom1->shape(i)==1)
              << "Dimensions must be equal or 1 in the bottoms!";
        top_shape[i] = bottom[0]->shape(i) >= bottom1->shape(i) ? bottom[0]->shape(i): bottom1->shape(i);
      }
    }
  }
  else if(dim_diff_ > 0) //bottom0 has more axes than bottom1
  {
    if(!is_scalar_)
    {
      for(int i=0;i<dim_diff_;i++)
        top_shape[i] = bottom[0]->shape(i);
      for(int i=dim_diff_; i<dim_; i++)
        top_shape[i] = bottom[0]->shape(i) >= bottom1->shape(i-dim_diff_) ? bottom[0]->shape(i): bottom1->shape(i-dim_diff_);
    }
    else //bottom1 is a scalar
    {
      for(int i=0;i<dim_;i++)
        top_shape[i] = bottom[0]->shape(i);
    }
  }
  else //dim_diff_<0, bottom1 has more axes than bottom0
  {
    if(!is_scalar_)
    {
      for(int i=0;i<-dim_diff_;i++)
        top_shape[i] = bottom1->shape(i);
      for(int i=-dim_diff_; i<dim_; i++)
        top_shape[i] = bottom[0]->shape(i+dim_diff_) >= bottom1->shape(i) ? bottom[0]->shape(i+dim_diff_): bottom1->shape(i);
    }
    else //bottom0 is a scalar
    {
      for(int i=0;i<dim_;i++)
        top_shape[i] = bottom1->shape(i);
    }
  }

  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PowLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	  const Dtype* bottom0_data = bottom[0]->cpu_data();
	  Blob<Dtype>* bottom1 = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
	  const Dtype* bottom1_data = bottom1->cpu_data();
	  Dtype* top_data = top[0]->mutable_cpu_data();
    int count = top[0]->count();

    // Assume top index (x,y,z) with top shape (A, B, C)
    // top offset d = xBC + yC + z
    // So to count the bottom index, should first figure out x, y, z
    // x = d / BC
    // y = (d % BC) / C
    // z = d % C
    // Then consider bottom shape (A', B', C'), while A = A' or 1
    // So bottom offset = x'B'C' + y'C' + z', while x' = x or 0
    if(!is_scalar_)
    {
      for(int d=0; d<count; d++)
      {
        int offset0 = 0;
        int offset1 = 0;

        if(dim_diff_ == 0)
        {
          for(int i=0;i<dim_-1;i++)
          {
            int num = (d % top[0]->count(i)) / top[0]->count(i+1);
            int n0 = 1 == bottom[0]->shape(i) ? 0 : num;
            int n1 = 1 == bottom1->shape(i) ? 0 : num;
            offset0 += n0 * bottom[0]->count(i+1);
            offset1 += n1 * bottom1->count(i+1);
          }
          int z = d % top[0]->shape(dim_-1);
          int z0 = 1 == bottom[0]->shape(dim_-1) ? 0 : z;
          int z1 = 1 == bottom1->shape(dim_-1) ? 0 : z;
          offset0 += z0;
          offset1 += z1;
        }
        else if(dim_diff_ > 0) //bottom0 has more axes than bottom1
        {
          for(int i=0;i<dim_diff_;i++)
          {
            int num = (d % top[0]->count(i)) / top[0]->count(i+1);
            int n0 = 1 == bottom[0]->shape(i) ? 0 : num;
            offset0 += n0 * bottom[0]->count(i+1);
          }
          for(int i=dim_diff_;i<dim_-1;i++)
          {
            int num = (d % top[0]->count(i)) / top[0]->count(i+1);
            int n0 = 1 == bottom[0]->shape(i) ? 0 : num;
            int n1 = 1 == bottom1->shape(i-dim_diff_) ? 0 : num;
            offset0 += n0 * bottom[0]->count(i+1);
            offset1 += n1 * bottom1->count(i-dim_diff_+1);
          }
          int z = d % top[0]->shape(dim_-1);
          int z0 = 1 == bottom[0]->shape(dim_-1) ? 0 : z;
          int z1 = 1 == bottom1->shape(dim_-dim_diff_-1) ? 0 : z;
          offset0 += z0;
          offset1 += z1;
        }
        else //dim_diff_<0, bottom1 has more axes than bottom0
        {
          for(int i=0;i<-dim_diff_;i++)
          {
            int num = (d % top[0]->count(i)) / top[0]->count(i+1);
            int n1 = 1 == bottom1->shape(i) ? 0 : num;
            offset1 += n1 * bottom1->count(i+1);
          }
          for(int i=-dim_diff_;i<dim_-1;i++)
          {
            int num = (d % top[0]->count(i)) / top[0]->count(i+1);
            int n0 = 1 == bottom[0]->shape(i+dim_diff_) ? 0 : num;
            int n1 = 1 == bottom1->shape(i) ? 0 : num;
            offset0 += n0 * bottom[0]->count(i+dim_diff_+1);
            offset1 += n1 * bottom1->count(i+1);
          }
          int z = d % top[0]->shape(dim_-1);
          int z0 = 1 == bottom[0]->shape(dim_+dim_diff_-1) ? 0 : z;
          int z1 = 1 == bottom1->shape(dim_-1) ? 0 : z;
          offset0 += z0;
          offset1 += z1;
        }

        top_data[d] = pow(bottom0_data[offset0], bottom1_data[offset1]);
        //caffe_powx(1, bottom0_data+offset0, bottom1_data[offset1], top_data);
      }
    }
    else //is scalar with shape ()
    {
      if(bottom1->num_axes()==0) //bottom1 is a scalar
      {
        Dtype scalar = bottom1_data[0];
        for(int d=0; d<count; d++)
        {
          top_data[d] = pow(bottom0_data[d], scalar);
        }
      }
      else //bottom0 is a scalar
      {
        Dtype scalar = bottom0_data[0];
        for(int d=0; d<count; d++)
        {
          top_data[d] = pow(scalar, bottom1_data[d]);
        }
      }
    }
}


INSTANTIATE_CLASS(PowLayer);
REGISTER_LAYER_CLASS(Pow);

}  // namespace caffe
