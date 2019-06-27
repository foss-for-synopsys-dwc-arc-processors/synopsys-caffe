#include <vector>

#include "caffe/layers/broadcast_to_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BroadcastToLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const BroadcastToParameter& broadcast_to_param = this->layer_param_.broadcast_to_param();
  output_shape_.clear();
  std::copy(broadcast_to_param.shape().begin(),
    broadcast_to_param.shape().end(),
    std::back_inserter(output_shape_));

  CHECK_GE(output_shape_.size(), bottom[0]->num_axes()) << "Output shape should not have less axis than input!";
  int dim_diff = output_shape_.size() - bottom[0]->num_axes();
  for(int i=output_shape_.size()-1; i>=dim_diff; i--)
  {
    CHECK_GT(output_shape_[i], 0) << "Values in output shape must be positive!";
    CHECK(output_shape_[i]==bottom[0]->shape(i-dim_diff) || bottom[0]->shape(i-dim_diff)==1)
        << "The broadcasting shape is incompatible with the input!";
  }
}


template <typename Dtype>
void BroadcastToLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(output_shape_);
}

template <typename Dtype>
void BroadcastToLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int count = top[0]->count();

  int dim = top[0]->num_axes();
  int dim_diff = output_shape_.size() - bottom[0]->num_axes();
  // Assume top index (x,y,z) with top shape (A, B, C)
  // top offset d = xBC + yC + z
  // So to count the bottom index, should first figure out x, y, z
  // x = d / BC
  // y = (d % BC) / C
  // z = d % C
  // Then consider bottom shape (A', B', C'), where A' = 1 or A
  // So bottom offset = x'B'C' + y'C' + z
  for(int d=0; d<count; d++)
  {
    int offset = 0;

    for(int i=dim_diff;i<dim-1;i++)
    {
      int num = (d % top[0]->count(i)) / top[0]->count(i+1);
      int n0 = 1 == bottom[0]->shape(i-dim_diff) ? 0 : num;
      offset += n0 * bottom[0]->count(i-dim_diff+1);
    }
    int z = d % top[0]->shape(dim-1);
    int z0 = 1 == bottom[0]->shape(dim-dim_diff-1) ? 0 : z;
    offset += z0;

    top_data[d] = bottom_data[offset];
  }
}

template <typename Dtype>
void BroadcastToLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

//#ifdef CPU_ONLY
//STUB_GPU(TileLayer);
//#endif

INSTANTIATE_CLASS(BroadcastToLayer);
REGISTER_LAYER_CLASS(BroadcastTo);

}  // namespace caffe
