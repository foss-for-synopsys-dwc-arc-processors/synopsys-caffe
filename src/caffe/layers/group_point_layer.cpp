#include <vector>
#include <typeinfo>

#include "caffe/layers/group_point_layer.hpp"


namespace caffe {

template <typename Dtype>
  void GroupPointLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
    // no need to setup params for this layer
    int num_axes_ = bottom[1]->num_axes();
    CHECK_LE(num_axes_, 3);
    CHECK_GE(num_axes_, 2);
  }

  template <typename Dtype>
  void GroupPointLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // bottom0 shape = (batch_size, num_point, num_cord=3)
    CHECK_EQ(bottom[0]->shape().size(), 3);
    // bottom1 shape = (batch_size, npoint, num_cord=3) if GatherPoint
    // bottom1 shape = (batch_size, npoint, nsample, num_cord=3) if GroupPoint
    // top0 shape = bottom1_shape.push_back(num_cord)
    vector<int> bottom_shape = bottom[1]->shape();
    CHECK_GE(bottom_shape.size(), 2);
    CHECK_LE(bottom_shape.size(), 3);
    bottom_shape.push_back(bottom[0]->shape(-1));
    top[0]->Reshape(bottom_shape);
  }


template <typename Dtype>
void GroupPointLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *indices_ = bottom[1]->cpu_data();
  // check indices_
  for (int i = 0; i < bottom[1]->count(); ++i) {
    CHECK_GE(indices_[i], 0)
        << "indices_ element with idx" << i << " is negative";
    CHECK_LT(indices_[i], bottom[0]->shape(1))
        << "indices_ element with idx" << i << " is out of range "
        << bottom[0]->shape(1);
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  const int B = bottom_shape[0];
  const int N = bottom_shape[1];
  const int D = bottom_shape[2];
  // num_axes doesn't matter; just skip the batch_dim of indices
  const int N_IDX = bottom[1]->count(1);
  const int gather_size_ = D;
  
  for(int b = 0; b < B; ++b){
    const Dtype *bot_ = bottom_data + b * N * gather_size_;
    const Dtype *idx_ = indices_ + b * N_IDX;
    Dtype *top_ = top_data + b * N_IDX * gather_size_;
    for(int i = 0; i < N_IDX; ++i) {
      // idx = indices_[b,i]
      // copy bottom_data[b, idx, :] to top_data[b, i, :]
      caffe_copy(gather_size_, bot_ + (int)idx_[i] * gather_size_, top_ + i * gather_size_);
    }
  }
}

INSTANTIATE_CLASS(GroupPointLayer);
REGISTER_LAYER_CLASS(GroupPoint);

}  // namespace caffe