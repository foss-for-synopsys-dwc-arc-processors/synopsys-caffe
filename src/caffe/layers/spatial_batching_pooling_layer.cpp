#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/spatial_batching_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SpatialBatchingPoolingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  SpatialBatchingPoolingParameter pool_param =
      this->layer_param_.spatial_batching_pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() || pool_param.has_kernel_h() ||
            pool_param.has_kernel_w()))
        << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
          !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
          (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h() &&
         pool_param.has_pad_w()) ||
        (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h() &&
         pool_param.has_stride_w()) ||
        (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  ceil_mode_ = pool_param.ceil_mode();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
        << "With Global_pooling: true; only pad = 0 and stride = 1";
  }

  saturate_ = pool_param.saturate();

  if (pool_param.has_pad_l()) {
    pad_l_ = pool_param.pad_l();
  } else {
    pad_l_ = 0;
  }
  if (pool_param.has_pad_r()) {
    pad_r_ = pool_param.pad_r();
  } else {
    pad_r_ = 0;
  }
  if (pool_param.has_pad_t()) {
    pad_t_ = pool_param.pad_t();
  } else {
    pad_t_ = 0;
  }
  if (pool_param.has_pad_b()) {
    pad_b_ = pool_param.pad_b();
  } else {
    pad_b_ = 0;
  }

  if (pad_l_ != 0 || pad_r_ != 0 || pad_t_ != 0 || pad_b_ != 0) {
    CHECK(!pool_param.has_pad())
        << "Either pad or pad_l/r/t/b should be specified; not both.";
    CHECK(!pool_param.has_pad_h() && !pool_param.has_pad_w())
        << "Either pad_h/w or pad_l/r/t/b should be specified; not both.";
  }
  // CUSTOMIZATION-->

  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.spatial_batching_pooling_param().pool() ==
              SpatialBatchingPoolingParameter_PoolMethod_AVE ||
          this->layer_param_.spatial_batching_pooling_param().pool() ==
              SpatialBatchingPoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }

  //<--CUSTOMIZATION for spatial batching
  if (pool_param.has_spatial_batching_h()) {
    spatial_batching_h_ = pool_param.spatial_batching_h();
  } else {
    spatial_batching_h_ = 0;
  }
  if (pool_param.has_spatial_batching_w()) {
    spatial_batching_w_ = pool_param.spatial_batching_w();
  } else {
    spatial_batching_w_ = 0;
  }
  if (pool_param.has_batch_h()) {
    batch_h_ = pool_param.batch_h();
  } else {
    batch_h_ = 0;
  }
  if (pool_param.has_batch_w()) {
    batch_w_ = pool_param.batch_w();
  } else {
    batch_w_ = 0;
  }
  if (pool_param.has_skip_h()) {
    skip_h_ = pool_param.skip_h();
  } else {
    skip_h_ = 0;
  }
  if (pool_param.has_skip_w()) {
    skip_w_ = pool_param.skip_w();
  } else {
    skip_w_ = 0;
  }
  // CUSTOMIZATION for spatial batching-->
}

template <typename Dtype>
void SpatialBatchingPoolingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(4, bottom[0]->num_axes())
      << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  // CUSTOMIZATION for spatial batching
  gap_h_ = -1;
  gap_w_ = -1;

  // Specify the structure by ceil or floor mode
  if (ceil_mode_) {
    if (pad_l_ != 0 || pad_r_ != 0 || pad_t_ != 0 || pad_b_ != 0) {
      pooled_height_ =
          static_cast<int>(
              ceil(static_cast<float>(height_ + pad_t_ + pad_b_ - kernel_h_) /
                   stride_h_)) +
          1;
      pooled_width_ =
          static_cast<int>(
              ceil(static_cast<float>(width_ + pad_l_ + pad_r_ - kernel_w_) /
                   stride_w_)) +
          1;
    } else {
      pooled_height_ =
          static_cast<int>(
              ceil(static_cast<float>(height_ + 2 * pad_h_ - kernel_h_) /
                   stride_h_)) +
          1;
      pooled_width_ = static_cast<int>(ceil(
                          static_cast<float>(width_ + 2 * pad_w_ - kernel_w_) /
                          stride_w_)) +
                      1;
    }
  } else {
    if (pad_l_ != 0 || pad_r_ != 0 || pad_t_ != 0 || pad_b_ != 0) {
      pooled_height_ =
          static_cast<int>(
              floor(static_cast<float>(height_ + pad_t_ + pad_b_ - kernel_h_ -
                                       (spatial_batching_h_ - 1) * skip_h_) /
                    stride_h_)) +
          1;
      pooled_width_ =
          static_cast<int>(
              floor(static_cast<float>(width_ + pad_l_ + pad_r_ - kernel_w_ -
                                       (spatial_batching_w_ - 1) * skip_w_) /
                    stride_w_)) +
          1;
      //<--CUSTOMIZATION for spatial batching
      if (spatial_batching_h_ != 0 && spatial_batching_w_ != 0 &&
          batch_h_ != 0 && batch_w_ != 0) {
        gap_h_ = static_cast<int>(
            floor(static_cast<float>(height_ - batch_h_ * spatial_batching_h_) /
                  (spatial_batching_h_ - 1)));
        gap_w_ = static_cast<int>(
            floor(static_cast<float>(width_ - batch_w_ * spatial_batching_w_) /
                  (spatial_batching_w_ - 1)));
        pooled_batch_h_ =
            static_cast<int>(floor(
                static_cast<float>(batch_h_ + pad_t_ + pad_b_ - kernel_h_) /
                stride_h_)) +
            1;
        pooled_batch_w_ =
            static_cast<int>(floor(
                static_cast<float>(batch_w_ + pad_l_ + pad_r_ - kernel_w_) /
                stride_w_)) +
            1;
        pooled_gap_h_ = static_cast<int>(
            floor(static_cast<float>(pooled_height_ -
                                     pooled_batch_h_ * spatial_batching_h_) /
                  (spatial_batching_h_ - 1)));
        pooled_gap_w_ = static_cast<int>(
            floor(static_cast<float>(pooled_width_ -
                                     pooled_batch_w_ * spatial_batching_w_) /
                  (spatial_batching_w_ - 1)));
      }
      // CUSTOMIZATION for spatial batching-->
    } else {
      pooled_height_ =
          static_cast<int>(
              floor(static_cast<float>(height_ + 2 * pad_h_ - kernel_h_) /
                    stride_h_)) +
          1;
      pooled_width_ = static_cast<int>(floor(
                          static_cast<float>(width_ + 2 * pad_w_ - kernel_w_) /
                          stride_w_)) +
                      1;
    }
  }

  if (pad_l_ || pad_r_ || pad_t_ || pad_b_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_t_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_l_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_t_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_l_);
  }
  // CUSTOMIZATION-->

  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }

  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.spatial_batching_pooling_param().pool() ==
          SpatialBatchingPoolingParameter_PoolMethod_MAX &&
      top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
                     pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void SpatialBatchingPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int *mask = NULL; // suppress warnings about uninitialized variables
  Dtype *top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.

  //<--CUSOMIZATION
  int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
  if (pad_l_ != 0 || pad_r_ != 0 || pad_t_ != 0 || pad_b_ != 0) {
    pad_top = pad_t_;
    pad_bottom = pad_b_;
    pad_left = pad_l_;
    pad_right = pad_r_;
  } else {
    pad_top = pad_h_;
    pad_bottom = pad_h_;
    pad_left = pad_w_;
    pad_right = pad_w_;
  }

  switch (this->layer_param_.spatial_batching_pooling_param().pool()) {
  case SpatialBatchingPoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int sbh = 0; sbh < spatial_batching_h_; sbh++) {
          int batch_hstart = sbh * (batch_h_ + gap_h_);
          int batch_hend = min(height_, (sbh + 1) * (batch_h_ + gap_h_));
          for (int ph = sbh * (pooled_batch_h_ + pooled_gap_h_);
               ph < min(pooled_height_,
                        (sbh + 1) * (pooled_batch_h_ + pooled_gap_h_));
               ++ph) {
            for (int sbw = 0; sbw < spatial_batching_w_; sbw++) {
              int batch_wstart = sbw * (batch_w_ + gap_w_);
              int batch_wend = min(width_, (sbw + 1) * (batch_w_ + gap_w_));
              for (int pw = sbw * (pooled_batch_w_ + pooled_gap_w_);
                   pw < min(pooled_width_,
                            (sbw + 1) * (pooled_batch_w_ + pooled_gap_w_));
                   ++pw) {
                int hstart = ph * stride_h_ - pad_top + skip_h_ * sbh;
                int wstart = pw * stride_w_ - pad_left + skip_w_ * sbw;
                // CUSTOMIZATION-->
                int hend = min(hstart + kernel_h_, batch_hend);
                int wend = min(wstart + kernel_w_, batch_wend);
                hstart = max(hstart, batch_hstart);
                wstart = max(wstart, batch_wstart);
                const int pool_index = ph * pooled_width_ + pw;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    const int index = h * width_ + w;
                    if (bottom_data[index] > top_data[pool_index]) {
                      top_data[pool_index] = bottom_data[index];
                      if (use_top_mask) {
                        top_mask[pool_index] = static_cast<Dtype>(index);
                      } else {
                        mask[pool_index] = index;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case SpatialBatchingPoolingParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case SpatialBatchingPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  caffe_cpu_saturate(top[0]->count(), top[0]->mutable_cpu_data(),
                     saturate_); // if None nothing happens
}

INSTANTIATE_CLASS(SpatialBatchingPoolingLayer);
REGISTER_LAYER_CLASS(SpatialBatchingPooling);

} // namespace caffe
