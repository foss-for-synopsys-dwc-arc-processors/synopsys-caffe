#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <numeric>

#include "caffe/layers/space_to_batch_nd_layer.hpp"
// implementation of https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
namespace caffe {
using namespace std;

template <typename Dtype>
void SpaceToBatchNDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SpaceToBatchNDParameter& space_to_batch_nd_param = this->layer_param_.space_to_batch_nd_param();
  auto bottom_shape = bottom[0]->shape();
  for(auto i : space_to_batch_nd_param.block_shape())
    block_shape_.push_back(i);
  for(auto i : space_to_batch_nd_param.paddings())
    paddings_.push_back(i);
  for (int i=0; i<block_shape_.size(); i++)
    if((bottom_shape[i+1] + paddings_[2*i] + paddings_[2*i+1]) % block_shape_[i] != 0){
      LOG(FATAL) << "block_shape[" << i << "] cannot divide bottom_shape[" << i+1 << "] + paddings[" << 2*i << "] + paddings[" << 2*i+1 << "]";
    }
}

template <typename Dtype>
void SpaceToBatchNDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  auto shape = bottom[0]->shape();
  for(auto i = 0; i < block_shape_.size(); i++){
    shape[0] *= block_shape_[i];
    shape[i+1] += paddings_[2*i] + paddings_[2*i+1];
    shape[i+1] /= block_shape_[i];
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
inline vector<int> SpaceToBatchNDLayer<Dtype>::indices(int offset, const vector<int> & shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for(int i = shape.size()-1; i>=0; i--){
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int SpaceToBatchNDLayer<Dtype>::offset(const vector<int>& indices, const vector<int> & shape) const {
  int offset = 0;
  for (int i = 0; i < shape.size(); ++i) {
    offset *= shape[i];
    offset += indices[i];
  }
  return offset;
}

template <typename Dtype>
void SpaceToBatchNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape = bottom_shape;

  // zero-pad spatical part
  int strides = 1;
  for(int i=block_shape_.size(); i<bottom_shape.size(); i++)
    strides *= bottom_shape[i];

  // 1. Zero-pad the start and end of dimensions [1, ..., M] of the input according to paddings to produce padded of shape padded_shape.
  for(int i=0; i<paddings_.size()/2; i++)
    top_shape[i+1] += paddings_[2*i] + paddings_[2*i+1];

  for(int position=0; position<bottom[0]->count()/strides; position++){
    vector<int> coord_bottom = indices(position*strides, bottom_shape);
    vector<int> coord_pad(coord_bottom);
    for(int i=0; i<paddings_.size()/2; i++)
      coord_pad[i+1] += paddings_[2*i];
    int position_top = offset(coord_pad, top_shape);
    copy_n(bottom_data+position*strides, strides, top_data+position_top);
  }
  // 2. Reshape padded to reshaped_padded of shape:
  // [batch] + [padded_shape[1] / block_shape[0], block_shape[0], ..., padded_shape[M] / block_shape[M-1], block_shape[M-1]] + remaining_shape
  vector<int> permuted_shape = top_shape;
  vector<int> permuted_order(2*top_shape.size()+1);
  iota(permuted_order.begin(), permuted_order.end(), 0);
  permuted_order[block_shape_.size()] = 0;
  for(int i=0, s=1; i<block_shape_.size(); i++){
    top_shape[i+s] /= block_shape_[i];
    permuted_shape[i+s] = top_shape[i+s];
    permuted_order[i+block_shape_.size()+1] = i + s;
    s++;
    top_shape.insert(top_shape.begin()+i+s, block_shape_[i]);
    permuted_shape.insert(permuted_shape.begin()+i, block_shape_[i]);
    permuted_order[i] = i + s;
  }

  // 3. Permute dimensions of reshaped_padded to produce permuted_reshaped_padded of shape:
  // block_shape + [batch] + [padded_shape[1] / block_shape[0], ..., padded_shape[M] / block_shape[M-1]] + remaining_shape
  vector<Dtype> top_temp(top[0]->count());
  copy_n(top_data, top[0]->count(), top_temp.begin());
  for(int i=0; i<top_temp.size(); i++){
    vector<int> coord_old = indices(i, top_shape);
    vector<int> coord_permuted = coord_old;
    for(int i=0; i<coord_old.size(); i++){
      coord_permuted[i] = coord_old[permuted_order[i]];
    }
    int position_permuted = offset(coord_permuted, permuted_shape);
    copy_n(top_temp.begin()+i, 1, top_data+position_permuted);
  }

  // 4. Reshape permuted_reshaped_padded to flatten block_shape into the batch dimension, producing an output tensor of shape:
  // [batch * prod(block_shape)] + [padded_shape[1] / block_shape[0], ..., padded_shape[M] / block_shape[M-1]] + remaining_shape
}

INSTANTIATE_CLASS(SpaceToBatchNDLayer);
REGISTER_LAYER_CLASS(SpaceToBatchND);

}  // namespace caffe
