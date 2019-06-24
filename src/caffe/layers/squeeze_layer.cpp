#include <vector>

#include "caffe/layers/squeeze_layer.hpp"

namespace caffe {

template <typename Dtype>
void SqueezeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  vector<int> axis;
  axis.clear();
  std::copy(this->layer_param_.squeeze_param().axis().begin(),
      this->layer_param_.squeeze_param().axis().end(),
      std::back_inserter(axis));
  int remove_all = false;
  //LOG(INFO)<<axis.size();

  if(!axis.empty())
  {
    for(int i=0; i<axis.size();i++)
    {
      CHECK_LE(axis[i], bottom[0]->num_axes())
            << "Given axis index should not be greater than bottom axis count!";
      axis[i] = bottom[0]->CanonicalAxisIndex(axis[i]);
      CHECK_EQ(bottom[0]->shape(axis[i]), 1) <<"Can't remove axis that is not shape 1!";
    }
  }
  else remove_all = true;

  vector<int> top_shape;

  for (int i = 0; i < bottom[0]->num_axes(); i++)
  {
    if(bottom[0]->shape(i)==1)
    {
      if(remove_all || std::find(axis.begin(), axis.end(), i) != axis.end())
        continue; //not add this shape=1 dimension in top shape
    }
    top_shape.push_back(bottom[0]->shape(i));
  }

  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype>
void SqueezeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void SqueezeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

INSTANTIATE_CLASS(SqueezeLayer);
REGISTER_LAYER_CLASS(Squeeze);

}  // namespace caffe
