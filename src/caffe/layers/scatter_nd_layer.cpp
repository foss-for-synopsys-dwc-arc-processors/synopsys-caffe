#include <vector>
#include <typeinfo>

#include "caffe/layers/scatter_nd_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScatterNDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  const ScatterNDParameter& scatter_nd_param = this->layer_param_.scatter_nd_param();
  is_data_param_ = scatter_nd_param.is_data_param();
  is_indices_param_ = scatter_nd_param.is_indices_param();
  is_updates_param_ = scatter_nd_param.is_updates_param();
  int blob_idx = 0, bottom_idx = 0, dims;
  vector<int> sz;
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else{
    this->blobs_.resize((int)is_data_param_ + is_indices_param_ + is_updates_param_);
    if (is_data_param_) {
      dims = scatter_nd_param.data_shape_size();
      sz.resize(dims);
      for (int i = 0; i < dims; ++i) {
        sz[i] = scatter_nd_param.data_shape(i);
      }
      this->blobs_[blob_idx++].reset(new Blob<Dtype>(sz));
    }
    if (is_indices_param_) {
      dims = scatter_nd_param.indices_shape_size();
      sz.resize(dims);
      for (int i = 0; i < dims; ++i) {
        sz[i] = scatter_nd_param.indices_shape(i);
      }
      this->blobs_[blob_idx++].reset(new Blob<Dtype>(sz));
    }
    if (is_updates_param_) {
      dims = scatter_nd_param.updates_shape_size();
      sz.resize(dims);
      for (int i = 0; i < dims; ++i) {
        sz[i] = scatter_nd_param.updates_shape(i);
      }
      this->blobs_[blob_idx++].reset(new Blob<Dtype>(sz));
    }
    blob_idx = 0;
  }
  
  vector<int> data_shape, indices_shape, updates_shape;
  if (is_data_param_) {
    data_shape = this->blobs_[blob_idx++]->shape();
  } else {
    data_shape = bottom[bottom_idx++]->shape();
  }
  if (is_indices_param_) {
    indices_shape = this->blobs_[blob_idx++]->shape();
  } else {
    indices_shape = bottom[bottom_idx++]->shape();
  }
  if (is_updates_param_) {
    updates_shape = this->blobs_[blob_idx++]->shape();
  } else {
    updates_shape = bottom[bottom_idx++]->shape();
  }
  
  CHECK_EQ(bottom_idx, bottom.size());
  CHECK_EQ(blob_idx, this->blobs_.size());
  CHECK_EQ(3, bottom_idx + blob_idx);

  Q_ = indices_shape.size(); // indices is rank Q, updates is rank (Q-1)+(R-K)
  K_ = indices_shape.back(); // a K-tuple in indices denotes a partial index into data
  const int R_ = data_shape.size();
  CHECK_LE(K_, R_);

  // assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]
  indices_shape.pop_back();
  for (int i = K_; i < R_; ++i) {
    indices_shape.push_back(data_shape[i]);
  }
  CHECK_EQ(updates_shape.size(), indices_shape.size());  
  for (int i = 0; i < indices_shape.size(); ++i){
    CHECK_EQ(updates_shape[i], indices_shape[i]);
  }
}

template <typename Dtype>
void ScatterNDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // output.shape = data.shape
  if (is_data_param_) top[0]->Reshape(this->blobs_[0]->shape());
  else top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void ScatterNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int blob_idx = 0, bottom_idx = 0;
  int num_update_, data_stride_, update_stride_, total_data_count_;
  vector<int> data_shape;
  const Dtype *bottom_data, *indices, *updates;
  if (is_data_param_) {
    data_shape = this->blobs_[blob_idx]->shape();
    data_stride_ = this->blobs_[blob_idx]->count(K_);
    total_data_count_ = this->blobs_[blob_idx]->count();
    bottom_data = this->blobs_[blob_idx++]->cpu_data();
  } else {
    data_shape = bottom[bottom_idx]->shape();
    data_stride_ = bottom[bottom_idx]->count(K_);
    total_data_count_ = bottom[bottom_idx]->count();
    bottom_data = bottom[bottom_idx++]->cpu_data();
  }
  if (is_indices_param_) {
    num_update_ = this->blobs_[blob_idx]->count(0,  Q_ - 1);
    indices = this->blobs_[blob_idx++]->cpu_data();
  } else {
    num_update_ = bottom[bottom_idx]->count(0,  Q_ - 1);
    indices = bottom[bottom_idx++]->cpu_data();
  }
  if (is_updates_param_) {
    update_stride_ = this->blobs_[blob_idx]->count(Q_ - 1);
    updates = this->blobs_[blob_idx++]->cpu_data();
  } else {
    update_stride_ = bottom[bottom_idx]->count(Q_ - 1);
    updates = bottom[bottom_idx++]->cpu_data();
  }

  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_copy(total_data_count_, bottom_data, top_data);

  int idx_ofs = 0;
  int upd_ofs = 0;
  for (int m = 0; m < num_update_; ++m) {
    // resolve indices[ofs:ofs+K] = (i_0, i_1, ..., i_{K-1})
    int idx = 0;
    for (int i = 0; i < K_; ++i){
      idx *= data_shape[i];
      idx += indices[idx_ofs + i];
    }
    idx *= data_stride_;
    // move updates[ofs:ofs+] to data[idx:idx+update_size]
    caffe_copy(update_stride_, updates + upd_ofs, top_data + idx);
    idx_ofs += K_;
    upd_ofs += update_stride_;
  }
}

INSTANTIATE_CLASS(ScatterNDLayer);
REGISTER_LAYER_CLASS(ScatterND);

}  // namespace caffe