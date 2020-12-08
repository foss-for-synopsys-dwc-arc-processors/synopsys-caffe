#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/icnet_subgraph_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ICNetSubgraphLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // the parameter is taken from icnet_cityscapes.prototxt layer conv5_3 to conv5_3_sum subgraph
  kernel_h_.push_back(33);
  kernel_h_.push_back(17);
  kernel_h_.push_back(13);
  kernel_h_.push_back(8);

  kernel_w_.push_back(65);
  kernel_w_.push_back(33);
  kernel_w_.push_back(25);
  kernel_w_.push_back(15);

  stride_h_.push_back(33);
  stride_h_.push_back(16);
  stride_h_.push_back(10);
  stride_h_.push_back(5);

  stride_w_.push_back(65);
  stride_w_.push_back(32);
  stride_w_.push_back(20);
  stride_w_.push_back(10);
}

template <typename Dtype>
void ICNetSubgraphLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  for (int i = 0; i < pool_branches_; i++)
  {
    pooled_height_.push_back(static_cast<int>(ceil(static_cast<float>(
        height_ + 2 * pad_h_ - kernel_h_[i]) / stride_h_[i])) + 1);
    pooled_width_.push_back(static_cast<int>(ceil(static_cast<float>(
        width_ + 2 * pad_w_ - kernel_w_[i]) / stride_w_[i])) + 1);
  }

  top[0]->ReshapeLike(*bottom[0]);

  // allocate for the largest size needed
  pooling_.Reshape(num_, channels_, pooled_height_[pool_branches_-1],
      pooled_width_[pool_branches_-1]);

  interp_.ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void ICNetSubgraphLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  // 1 branch data directly copied to final accumulator
  for (int i = 0; i < top_count; ++i) {
    top_data[i] = bottom_data[i];
  }

  int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;

  // The main loop
  for (int i = 0; i < pool_branches_; i++)
  {
    bottom_data = bottom[0]->cpu_data(); // reset the pointer
    // do the ave pooling
    Dtype* pool_data = pooling_.mutable_cpu_data();
    int pool_count = pooling_.count();
    for (int j = 0; j < pool_count; ++j) {
      pool_data[j] = 0; // clear the field
    }

    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_[i]; ++ph) {
          for (int pw = 0; pw < pooled_width_[i]; ++pw) {
            int hstart = ph * stride_h_[i] - pad_top;
            int wstart = pw * stride_w_[i] - pad_left;
            int hend = min(hstart + kernel_h_[i], height_ + pad_bottom);
            int wend = min(wstart + kernel_w_[i], width_ + pad_right);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                pool_data[ph * pooled_width_[i] + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            pool_data[ph * pooled_width_[i] + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        pool_data += pooled_height_[i] * pooled_width_[i];
      }
    }

    // do the interp computation
    int pad_beg_ = 0;
    int pad_end_ = 0;
    int height_in_ = pooled_height_[i];
    int width_in_ = pooled_width_[i];
    int height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
    int width_in_eff_ = width_in_ + pad_beg_ + pad_end_;

    Dtype* interp_data = interp_.mutable_cpu_data();
    int interp_count = interp_.count();
    for (int j = 0; j < interp_count; ++j) {
      interp_data[j] = 0; // clear the field
    }

    caffe_cpu_interp2<Dtype,false>(num_ * channels_,
        pooling_.cpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
        interp_.mutable_cpu_data(), 0, 0, height_, width_, height_, width_);

    // accumulate
    for (int j = 0; j < top_count; ++j) {
      top_data[j] += interp_data[j];
    }
  }
}

INSTANTIATE_CLASS(ICNetSubgraphLayer);
REGISTER_LAYER_CLASS(ICNetSubgraph);

}  // namespace caffe
