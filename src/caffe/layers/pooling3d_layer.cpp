#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void Pooling3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  Pooling3DParameter pool3d_param = this->layer_param_.pooling3d_param();
  global_pooling_ = pool3d_param.global_pooling();
  ceil_mode_ = pool3d_param.ceil_mode();

  CHECK_EQ(bottom[0]->num_axes(), 5) << "Input must have 5 axes.";

  if (global_pooling_) {
    // TODO: only support data_format in NCDHW now
    kernel_d_ = bottom[0]->shape(2);
    kernel_h_ = bottom[0]->shape(3);
    kernel_w_ = bottom[0]->shape(4);
  } else {
    if (pool3d_param.has_kernel_size())
      kernel_d_ = kernel_h_ = kernel_w_ = pool3d_param.kernel_size();
    else {
      kernel_d_ = pool3d_param.kernel_d();
      kernel_h_ = pool3d_param.kernel_h();
      kernel_w_ = pool3d_param.kernel_w();
    }
  }

  if (pool3d_param.has_stride())
    stride_d_ = stride_h_ = stride_w_ = pool3d_param.stride();
  else {
    stride_d_ = pool3d_param.stride_d();
    stride_h_ = pool3d_param.stride_h();
    stride_w_ = pool3d_param.stride_w();
  }

  if (pool3d_param.has_pad())
    pad_h0_ = pad_w0_ = pad_d0_ = pad_h1_ = pad_w1_ = pad_d1_ = pool3d_param.pad();
  else {
    pad_h0_ = pool3d_param.pad_h0();
    pad_h1_ = pool3d_param.pad_h1();
    pad_w0_ = pool3d_param.pad_w0();
    pad_w1_ = pool3d_param.pad_w1();
    pad_d0_ = pool3d_param.pad_d0();
    pad_d1_ = pool3d_param.pad_d1();
  }
}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  // TODO: only support data_format in NCDHW now
  num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  depth_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);

  if (ceil_mode_) {
    pooled_height_ = static_cast<int>(ceil(static_cast<float>(
        height_ + pad_h0_ + pad_h1_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(ceil(static_cast<float>(
        width_ + pad_w0_ + pad_w1_ - kernel_w_) / stride_w_)) + 1;
    pooled_depth_ = static_cast<int>(ceil(static_cast<float>(
        depth_ + pad_d0_ + pad_d1_ - kernel_d_) / stride_d_)) + 1;
  } else {
    pooled_height_ = static_cast<int>(floor(static_cast<float>(
        height_ + pad_h0_ + pad_h1_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(floor(static_cast<float>(
        width_ + pad_w0_ + pad_w1_ - kernel_w_) / stride_w_)) + 1;
    pooled_depth_ = static_cast<int>(floor(static_cast<float>(
        depth_ + pad_d0_ + pad_d1_ - kernel_d_) / stride_d_)) + 1;
  }

  if (pad_h0_ || pad_h1_ || pad_w0_ || pad_w1_ || pad_d0_ || pad_d1_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h0_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w0_) {
      --pooled_width_;
    }
    if ((pooled_depth_ - 1) * stride_d_ >= depth_ + pad_d0_) {
      --pooled_depth_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h0_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w0_);
    CHECK_LT((pooled_depth_ - 1) * stride_d_, depth_ + pad_d0_);
  }

  vector<int> shape(5);
  // TODO: only support data_format in NCDHW now
  shape[0] = num_;
  shape[1] = channels_;
  shape[2] = pooled_depth_;
  shape[3] = pooled_height_;
  shape[4] = pooled_width_;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  switch (this->layer_param_.pooling3d_param().pool()) {
    case Pooling3DParameter_PoolMethod_MAX:
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);
      // The main loop
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int pd = 0; pd < pooled_depth_; ++pd) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int dstart = pd * stride_d_ - pad_d0_;
                int hstart = ph * stride_h_ - pad_h0_;
                int wstart = pw * stride_w_ - pad_w0_;
                int dend = min(dstart + kernel_d_, depth_);
                int hend = min(hstart + kernel_h_, height_);
                int wend = min(wstart + kernel_w_, width_);
                dstart = max(dstart, 0);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                const int pool_index = pd * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw;
                for (int d = dstart; d < dend; ++d) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      const int index = d * height_ * width_ + h * width_ + w;
                      if (bottom_data[index] > top_data[pool_index]) {
                        top_data[pool_index] = bottom_data[index];

                      }
                    }
                  }
                }
              }
            }
          }
          // compute offset(n, c, d, h, w): (((n * channels() + c) * depth + d) * height() + h) * width() + w
          // (0, 1, 0, 0, 0)
          int bottom_offset = depth_ * height_ * width_;
          int top_offset = pooled_depth_ *  pooled_height_ * pooled_width_;
          bottom_data += bottom_offset;
          top_data += top_offset;
        }
      }
      break;
    case Pooling3DParameter_PoolMethod_AVE:
      for (int i = 0; i < top_count; ++i) {
        top_data[i] = 0;
      }

      break;
    case Pooling3DParameter_PoolMethod_AVE_EXC_PAD:
       for (int i = 0; i < top_count; ++i) {
         top_data[i] = 0;
       }

       break;
    default:
        LOG(FATAL) << "Unknown pooling method.";
  }
}

INSTANTIATE_CLASS(Pooling3DLayer);
REGISTER_LAYER_CLASS(Pooling3D);

} // namespace caffe
