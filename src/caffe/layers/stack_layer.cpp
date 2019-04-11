#include <vector>

#include "caffe/layers/stack_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StackLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void StackLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const StackParameter& stack_param = this->layer_param_.stack_param();

	// axis range [-(r+1), r+1)
  const int axis = stack_param.axis();
	if (axis < 0) {
		CHECK_GE(axis, -(num_axes + 1))
			<< "axis" << axis << "out of range for [" 
			<< -(num_axes + 1) << ", " << num_axes + 1 << ")";
		stack_axis_ = axis + num_axes + 1;
	}
	else {
		CHECK_LT(axis, num_axes+1) 
			<< "axis" << axis << "out of range for ["
			<< -(num_axes + 1) << ", " << num_axes + 1 << ")";
		stack_axis_ = axis;
	}

  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
	vector<int> bottom_shape = bottom[0]->shape();
  num_stack_ = bottom[0]->count(0, stack_axis_);
  stack_size_ = bottom[0]->count(stack_axis_);
	int bottom_count_sum = bottom[0]->count();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape";
    }
    bottom_count_sum += bottom[i]->count();
  }
	top_shape.resize(num_axes + 1);
	const int n_inputs = bottom.size();
	top_shape[stack_axis_] = n_inputs;
	for (int i = stack_axis_; i < num_axes; ++i) {
		top_shape[i + 1] = bottom_shape[i];
	}
  top[0]->Reshape(top_shape);
	// check the count
  CHECK_EQ(bottom_count_sum, top[0]->count());
	// case: only one input 
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void StackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// case: only one input
  if (bottom.size() == 1) { return; }

  Dtype* top_data = top[0]->mutable_cpu_data();
	const int n_inputs = bottom.size();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    for (int n = 0; n < num_stack_; ++n) {
      caffe_copy(stack_size_,
          bottom_data + n * stack_size_,
          top_data + (n * n_inputs + i) * stack_size_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(StackLayer);
#endif

INSTANTIATE_CLASS(StackLayer);
REGISTER_LAYER_CLASS(Stack);

}  // namespace caffe

