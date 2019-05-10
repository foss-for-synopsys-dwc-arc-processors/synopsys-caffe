#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <numeric>

#include "caffe/layers/batch_to_space_nd_layer.hpp"
// implementation of https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
namespace caffe {
using namespace std;

template <typename Dtype>
void BatchToSpaceNDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BatchToSpaceNDParameter& batch_to_space_nd_param = this->layer_param_.batch_to_space_nd_param();
  for(auto i : batch_to_space_nd_param.block_shape())
    block_shape_.push_back(i);
  for(auto i : batch_to_space_nd_param.crops())
    crops_.push_back(i);
}

template <typename Dtype>
void BatchToSpaceNDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  auto shape = bottom[0]->shape();
  for(auto i = 0; i < block_shape_.size(); i++){
    shape[0] /= block_shape_[i];
    shape[i+1] *= block_shape_[i];
    shape[i+1] -= crops_[2*i] + crops_[2*i+1];
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
inline vector<int> BatchToSpaceNDLayer<Dtype>::indices(int offset, const vector<int> & shape) const {
  vector<int> indices(shape.size());
  int r = offset;
  for(int i = shape.size()-1; i>=0; i--){
    indices[i] = r % shape[i];
    r /= shape[i];
  }
  return indices;
}

template <typename Dtype>
inline int BatchToSpaceNDLayer<Dtype>::offset(const vector<int>& indices, const vector<int> & shape) const {
  int offset = 0;
  for (int i = 0; i < shape.size(); ++i) {
    offset *= shape[i];
    offset += indices[i];
  }
  return offset;
}

template <typename Dtype>
void BatchToSpaceNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  // 1. Reshape input to reshaped of shape:
  // [block_shape[0], ..., block_shape[M-1], batch / prod(block_shape), input_shape[1], ..., input_shape[N-1]]
  // Permute dimensions of reshaped_padded to produce permuted_reshaped_padded of shape:
  // block_shape + [batch] + [padded_shape[1] / block_shape[0], ..., padded_shape[M] / block_shape[M-1]] + remaining_shape
  vector<Dtype> bottom_temp(bottom[0]->count());
  vector<int> bottom_temp_shape = bottom_shape;

  bottom_temp_shape.insert(bottom_temp_shape.begin(), block_shape_.begin(), block_shape_.end());
  for(auto i : block_shape_)
    bottom_temp_shape[block_shape_.size()] /= i;
  // 2. Permute dimensions of reshaped to produce permuted of shape [batch / prod(block_shape),
  // input_shape[1], block_shape[0], ..., input_shape[M], block_shape[M-1],
  // input_shape[M+1], ..., input_shape[N-1]]
  vector<int> permuted_shape = bottom_temp_shape;
  vector<int> permuted_order(bottom_temp_shape.size());
  iota(permuted_order.begin(), permuted_order.end(), 0);
  for(int i=0; i<block_shape_.size(); i++){
    permuted_shape[2*i+1] = bottom_temp_shape[i+block_shape_.size()+1];
    permuted_shape[2*i+2] = bottom_temp_shape[i];
    permuted_order[2*i+1] = i + block_shape_.size() + 1;
    permuted_order[2*i+2] = i;
  }
  permuted_order[0] = block_shape_.size();
  permuted_shape[0] = bottom_temp_shape[permuted_order[0]];

  int strides = 1;
  for(int i=2*block_shape_.size()+1; i<bottom_temp_shape.size(); i++)
    strides *= bottom_temp_shape[i];

  for(int position=0; position<bottom[0]->count()/strides; position++){
    vector<int> coord_bottom = indices(position*strides, bottom_temp_shape);
    vector<int> coord_permuted(coord_bottom);
    for(int i=0; i<bottom_temp_shape.size(); i++)
      coord_permuted[i] = coord_bottom[permuted_order[i]];
    int position_permuted = offset(coord_permuted, permuted_shape);
    copy_n(bottom_data+position*strides, strides, bottom_temp.begin()+position_permuted);
  }
  // 3. Reshape permuted to produce reshaped_permuted of shape [batch / prod(block_shape),
  // input_shape[1] * block_shape[0], ..., input_shape[M] * block_shape[M-1],
  // input_shape[M+1], ..., input_shape[N-1]]
  for(int i=0; i<block_shape_.size(); i++){
    permuted_shape[1+i] *= permuted_shape[2+i];
    permuted_shape.erase(permuted_shape.begin()+2+i, permuted_shape.begin()+3+i);
  }
  // input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1], ..., input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
  // input_shape[M+1], ..., input_shape[N-1]]
  for(int i=0; i<top[0]->count(); i++){
    vector<int> coord_top = indices(i, top[0]->shape());
    vector<int> coord_cropped = coord_top;
    for(int i=0; i<crops_.size()/2; i++){
      coord_cropped[i+1] += crops_[2*i];
    }
    int position_cropped = offset(coord_cropped, permuted_shape);
    top_data[i] = bottom_temp[position_cropped];
    // copy_n(bottom_temp.begin()+position_cropped, 1, top_data+i);
  }

}

INSTANTIATE_CLASS(BatchToSpaceNDLayer);
REGISTER_LAYER_CLASS(BatchToSpaceND);

}  // namespace caffe
