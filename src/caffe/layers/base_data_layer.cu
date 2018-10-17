#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    if(this->box_label_) {
      for (int j = 0; j < top.size() - 1; ++j) {
        top[j+1]->ReshapeLike(*(prefetch_current_->multi_label_[j]));
        top[j+1]->set_gpu_data(prefetch_current_->multi_label_[j]->mutable_gpu_data());
      }
    }
    else {
      // Reshape to loaded labels.
      top[1]->ReshapeLike(prefetch_current_->label_);
      top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
    }
  }
}

template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch =
    BasePrefetchingDataLayer<Dtype>::prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  if (output_data_dim_) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(batch->dim_);
    // Copy the labels.
    caffe_copy(batch->dim_.count(), batch->dim_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  BasePrefetchingDataLayer<Dtype>::prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ImageDimPrefetchingDataLayer);

}  // namespace caffe
