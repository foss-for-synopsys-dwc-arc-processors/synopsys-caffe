#include <algorithm>
#include <vector>

#include "caffe/layers/unstack_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void UnstackLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const UnstackParameter& unstack_param = this->layer_param_.unstack_param();
  unstack_axis_ = bottom[0]->CanonicalAxisIndex(unstack_param.axis());
  const int num = unstack_param.num();
	if (num != 0) {
		CHECK_EQ(num, bottom[0]->shape(unstack_axis_))
			<< "num should equal to the shape in axis!";
	}
  unstack_num_ = bottom[0]->shape(unstack_axis_);
  CHECK_EQ(unstack_num_, top.size())<< "Number of top blobs (" 
			<< top.size() << ") should euqal to "<< "shape in axis (" 
			<< unstack_num_ << ")";
}

template <typename Dtype>
void UnstackLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  //const UnstackParameter& unstack_param = this->layer_param_.unstack_param();
	vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(num_axes - 1);
  num_unstack_ = bottom[0]->count(0, unstack_axis_);
  unstack_size_ = bottom[0]->count(unstack_axis_ + 1);
	int count = 0;
	for (int i = unstack_axis_; i < num_axes - 1; ++i) {
		top_shape[i] = bottom_shape[i + 1];
	}
  for (int i = 0; i < top.size(); ++i) {
		top[i]->Reshape(top_shape);
		count += top[i]->count();
  }
  CHECK_EQ(count, bottom[0]->count());
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void UnstackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
  int offset_unstack_axis = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int bottom_unstack_axis = bottom[0]->shape(unstack_axis_);
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < num_unstack_; ++n) {
      const int top_offset = n * unstack_size_;
      const int bottom_offset =
          (n * bottom_unstack_axis + offset_unstack_axis) * unstack_size_;
      caffe_copy(unstack_size_,
          bottom_data + bottom_offset, top_data + top_offset);
    }
    offset_unstack_axis += 1;
  }
}



INSTANTIATE_CLASS(UnstackLayer);
REGISTER_LAYER_CLASS(Unstack);

}  // namespace caffe

