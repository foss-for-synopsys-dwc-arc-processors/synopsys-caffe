#include <vector>

#include "caffe/layers/expand_layer.hpp"

namespace caffe {

template <typename Dtype>
void ExpandLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  int axis = this->layer_param_.expand_param().axis();
  CHECK_LE(axis, bottom[0]->num_axes())
      << "Newly inserted axis index should not be greater than bottom axis count!";
  if(axis != bottom[0]->num_axes())
    axis = bottom[0]->CanonicalAxisIndex(axis);
  vector<int> top_shape;
  for (int i = 0; i < axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top_shape.push_back(1);
  for (int i = axis; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype>
void ExpandLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void ExpandLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

INSTANTIATE_CLASS(ExpandLayer);
REGISTER_LAYER_CLASS(Expand);

}  // namespace caffe
