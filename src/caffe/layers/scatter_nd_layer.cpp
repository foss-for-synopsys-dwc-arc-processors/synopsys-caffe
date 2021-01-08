#include <vector>
#include <typeinfo>

#include "caffe/layers/scatter_nd_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScatterNDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  vector<int> data_shape = bottom[0]->shape();
  vector<int> indices_shape = bottom[1]->shape();
  vector<int> updates_shape = bottom[2]->shape();
  const int K = bottom[1]->shape(-1);
  const int data_dims = bottom[0]->num_axes();
  CHECK_LE(K, data_dims);
  // assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]
  const int indices_dims = bottom[1]->num_axes();
  indices_shape.pop_back();
  for (int i = K; i < data_dims; ++i){
    indices_shape.push_back(data_shape[i]);
  }
  printf("%d v.s. %d\n", updates_shape.size(), indices_shape.size());
  for (int i = 0; i < indices_shape.size(); ++i){
    CHECK_EQ(updates_shape[i], indices_shape[i]);
  }
}

template <typename Dtype>
void ScatterNDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // data, indices, updates
  vector<int> bottom_shape = bottom[0]->shape();
  top[0]->Reshape(bottom_shape);
}

template <typename Dtype>
void ScatterNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_update_ = bottom[1]->count(0,  bottom[1]->num_axes() - 1);
  const vector<int> data_shape = bottom[0]->shape();
  //const int R_ = bottom[0]->num_axes(); // data is rank R
  const int Q_ = bottom[1]->num_axes(); // indices is rank Q, updates is rank (Q-1)+(R-K)
  const int K_ = bottom[1]->shape(-1); // a K-tuple in indices denotes a partial index into data
  const int update_stride_ = bottom[2]->count(Q_ - 1);
  //const int update_size_ = bottom[2]->count(Q_);
  const int data_stride_ = bottom[0]->count(K_);
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *indices = bottom[1]->cpu_data();
  const Dtype *updates = bottom[2]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);

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