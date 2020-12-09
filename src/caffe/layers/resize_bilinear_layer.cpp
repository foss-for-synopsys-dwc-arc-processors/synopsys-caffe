#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/layers/resize_bilinear_layer.hpp"

namespace caffe {

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  align_corners_ = this->layer_param_.resize_bilinear_param().align_corners();
  data_format_ = this->layer_param_.resize_bilinear_param().data_format();
  output_height_ = this->layer_param_.resize_bilinear_param().output_height();
  output_width_ = this->layer_param_.resize_bilinear_param().output_width();
  half_pixel_centers_ = this->layer_param_.resize_bilinear_param().half_pixel_centers();
  CHECK(!(align_corners_ && half_pixel_centers_)) <<
        "If half_pixel_centers is True, align_corners must be False.";
  pytorch_half_pixel_ = this->layer_param_.resize_bilinear_param().pytorch_half_pixel();
  CHECK(!(align_corners_ && pytorch_half_pixel_)) <<
        "If pytorch_half_pixel_ is True, align_corners must be False.";
  CHECK(!(half_pixel_centers_ && pytorch_half_pixel_)) <<
        "If pytorch_half_pixel_ is True, half_pixel_centers_ must be False.";
}

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  const int batch_size = bottom_shape[0];
  if(data_format_ == "NHWC"){
    const int channels = bottom_shape[3];
    if(!this->layer_param_.resize_bilinear_param().has_output_height())
    {
      float scale_height = this->layer_param_.resize_bilinear_param().scale_height();
      output_height_ = floor(bottom_shape[1] * scale_height);
    }
    if(!this->layer_param_.resize_bilinear_param().has_output_width())
    {
      float scale_width = this->layer_param_.resize_bilinear_param().scale_width();
      output_width_ = floor(bottom_shape[2] * scale_width);
    }
    top[0]->Reshape(batch_size, output_height_, output_width_, channels);
  } else { //NCHW
    const int channels = bottom_shape[1];
    if(!this->layer_param_.resize_bilinear_param().has_output_height())
    {
      float scale_height = this->layer_param_.resize_bilinear_param().scale_height();
      output_height_ = floor(bottom_shape[2] * scale_height);
    }
    if(!this->layer_param_.resize_bilinear_param().has_output_width())
    {
      float scale_width = this->layer_param_.resize_bilinear_param().scale_width();
      output_width_ = floor(bottom_shape[3] * scale_width);
    }
    top[0]->Reshape(batch_size, channels, output_height_, output_width_);
  }
}

// CalculateResizeScale computes the float scaling factor for height and width.
inline float CalculateResizeScale(int out_size, int in_size, bool align_corners) {
  return ( (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size) );
}

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::compute_interpolation_weights(const int out_size,
                                          const int in_size,
                                          const float scale,
                                          struct CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int i = out_size - 1; i >= 0; --i) {
    float in;
    if (align_corners_)
    {
      in = static_cast<float>(i) * scale;
    }
    else if (pytorch_half_pixel_) {
      // ref: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
      // https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/resize.py#L132
      if (out_size > 1) {
        in = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
      } else {
        in = -0.5f;
      }
    }
    else if (half_pixel_centers_)
    {
      in = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
      // ref: https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/core/kernels/image_resizer_state.h#L50
    }
    else { // no coordinate_transformation_mode required
      in = static_cast<float>(i) * scale;
    }
    const float in_f = std::floor(in);
    interpolation[i].lower =
        std::max(static_cast<int>(in_f), static_cast<int>(0));
    interpolation[i].upper =
        std::min(static_cast<int>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape = top[0]->shape();
  const int batch_size = bottom_shape[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  if(data_format_ == "NHWC"){
    const int input_height = bottom_shape[1];
    const int input_width = bottom_shape[2];
    const int channels = bottom_shape[3];

    std::vector<CachedInterpolation> ys(output_height_ + 1);
    std::vector<CachedInterpolation> xs(output_width_ + 1);

    const float height_scale =
      CalculateResizeScale(output_height_, input_height, align_corners_);
    const float width_scale =
      CalculateResizeScale(output_width_, input_width, align_corners_);

    compute_interpolation_weights(output_height_, input_height,
                                  height_scale, ys.data());
    compute_interpolation_weights(output_width_, input_width,
                                  width_scale, xs.data());
    // Scale x interpolation weights to avoid a multiplication during iteration.
    for (int i = 0; i < xs.size(); ++i) {
      xs[i].lower *= channels;
      xs[i].upper *= channels;
    }

    const int in_row_size = input_width * channels;
    const int in_batch_num_values = input_height * in_row_size;
    const int out_row_size = output_width_ * channels;

    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < output_height_; ++y) {
        const Dtype* ys_input_lower_ptr = bottom_data + ys[y].lower * in_row_size;
        const Dtype* ys_input_upper_ptr = bottom_data + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int x = 0; x < output_width_; ++x) {
          int xs_lower = xs[x].lower;
          int xs_upper = xs[x].upper;
          float xs_lerp = xs[x].lerp;
          for (int c = 0; c < channels; ++c) {
            const float top_left(ys_input_lower_ptr[xs_lower + c]);
            const float top_right(ys_input_lower_ptr[xs_upper + c]);
            const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
            const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
            top_data[x * channels + c] =
                compute_lerp(top_left, top_right, bottom_left, bottom_right,
                             xs_lerp, ys_lerp);
          }
        }
        top_data += out_row_size;
      }
      bottom_data += in_batch_num_values;
    }
  }
  else{  //NCHW
    const int channels = bottom_shape[1];
    const int input_height = bottom_shape[2];
    const int input_width = bottom_shape[3];

    std::vector<CachedInterpolation> ys(output_height_ + 1);
    std::vector<CachedInterpolation> xs(output_width_ + 1);

    const float height_scale =
      CalculateResizeScale(output_height_, input_height, align_corners_);
    const float width_scale =
      CalculateResizeScale(output_width_, input_width, align_corners_);

    compute_interpolation_weights(output_height_, input_height,
                                  height_scale, ys.data());
    compute_interpolation_weights(output_width_, input_width,
                                  width_scale, xs.data());
    // Scale x interpolation weights to avoid a multiplication during iteration.
    //for (int i = 0; i < xs.size(); ++i) {
    //  xs[i].lower *= channels;
    //  xs[i].upper *= channels;
    //}

    const int in_row_size = input_width;
    const int in_channel_num_values = input_height * in_row_size; //
    const int out_row_size = output_width_;

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < output_height_; ++y) {
          const Dtype* ys_input_lower_ptr = bottom_data + ys[y].lower * in_row_size;
          const Dtype* ys_input_upper_ptr = bottom_data + ys[y].upper * in_row_size;
          const float ys_lerp = ys[y].lerp;
          for (int x = 0; x < output_width_; ++x) {
            int xs_lower = xs[x].lower;
            int xs_upper = xs[x].upper;
            float xs_lerp = xs[x].lerp;
            const float top_left(ys_input_lower_ptr[xs_lower]);
            const float top_right(ys_input_lower_ptr[xs_upper]);
            const float bottom_left(ys_input_upper_ptr[xs_lower]);
            const float bottom_right(ys_input_upper_ptr[xs_upper]);
            top_data[x] =
                compute_lerp(top_left, top_right, bottom_left, bottom_right,
                    xs_lerp, ys_lerp);
          }
          top_data += out_row_size;
        }
        bottom_data += in_channel_num_values;
      }
    }
  }
}

INSTANTIATE_CLASS(ResizeBilinearLayer);
REGISTER_LAYER_CLASS(ResizeBilinear);

}  // namespace caffe
