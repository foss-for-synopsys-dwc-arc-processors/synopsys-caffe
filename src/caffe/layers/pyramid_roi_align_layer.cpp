#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/pyramid_roi_align_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PyramidROIAlignLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const PyramidROIAlignParameter &pyramid_roi_align_param =
      this->layer_param_.pyramid_roi_align_param();
  crop_height_ = pyramid_roi_align_param.crop_height();
  crop_width_ = pyramid_roi_align_param.crop_width();
  data_format_ = pyramid_roi_align_param.data_format();
  CHECK_GT(crop_height_, 0) << "crop_height must be > 0";
  CHECK_GT(crop_width_, 0) << "crop_width must be > 0";
  extrapolation_value_ = pyramid_roi_align_param.extrapolation_value();
}

template <typename Dtype>
void PyramidROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape {bottom[0]->shape(0), bottom[0]->shape(1), 0, 0, 0};
  if (data_format_ == "NHWC"){
    top_shape[2] = crop_height_;
    top_shape[3] = crop_width_;
    top_shape[4] = bottom[2]->shape(-1);
  } else {
    top_shape[2] = bottom[2]->shape(-1);
    top_shape[3] = crop_height_;
    top_shape[4] = crop_width_;
  }

  top[0]->Reshape(top_shape);
}


// https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
template <typename Dtype>
inline void PyramidROIAlignLayer<Dtype>::crop_and_resize(
    const Dtype *image, const Dtype *box, Dtype *top_data,
    const int image_height_, const int image_width_,
    const int channels_, const string data_format_) {
  const float y1 = box[0];
  const float x1 = box[1];
  const float y2 = box[2];
  const float x2 = box[3];

  const float height_scale =
    (crop_height_ > 1)
    ? (y2 - y1) * (image_height_ - 1) / (crop_height_ - 1)
    : 0;
  const float width_scale =
    (crop_width_ > 1) ? (x2 - x1) * (image_width_ - 1) / (crop_width_ - 1)
    : 0;

  if (data_format_ == "NHWC") {
    for (int y = 0; y < crop_height_; ++y) {
      const float in_y = (crop_height_ > 1)
        ? y1 * (image_height_ - 1) + y * height_scale
        : 0.5 * (y1 + y2) * (image_height_ - 1);
      if (in_y < 0 || in_y > image_height_ - 1) {
        for (int x = 0; x < crop_width_; ++x) {
          for (int d = 0; d < channels_; ++d) {
            top_data[(y * crop_width_ + x) * channels_ + d] = extrapolation_value_;
          }
        }
        continue;
      }

      const int top_y_index = floorf(in_y);
      const int bottom_y_index = ceilf(in_y);
      const float y_lerp = in_y - top_y_index;

      for (int x = 0; x < crop_width_; ++x) {
        const float in_x = (crop_width_ > 1)
          ? x1 * (image_width_ - 1) + x * width_scale
          : 0.5 * (x1 + x2) * (image_width_ - 1);
        if (in_x < 0 || in_x > image_width_ - 1) {
          for (int d = 0; d < channels_; ++d) {
            top_data[(y * crop_width_ + x) * channels_ + d] = extrapolation_value_;
          }
          continue;
        }
        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        for (int d = 0; d < channels_; ++d) {
          int index = (top_y_index * image_width_ + left_x_index) * channels_ + d;
          const float top_left = static_cast<float>(image[index]);

          index = (top_y_index * image_width_ + right_x_index) * channels_ + d;
          const float top_right = static_cast<float>(image[index]);

          index = (bottom_y_index * image_width_ + left_x_index) * channels_ + d;
          const float bottom_left = static_cast<float>(image[index]);

          index = (bottom_y_index * image_width_ + right_x_index) * channels_ + d;
          const float bottom_right = static_cast<float>(image[index]);

          const float top = top_left + (top_right - top_left) * x_lerp;
          const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

          top_data[(y * crop_width_ + x) * channels_ + d] = top + (bottom - top) * y_lerp;
        }
      }
    }
  } else { // NCHW format
    for (int d = 0; d < channels_; ++d) {
      for (int y = 0; y < crop_height_; ++y) {
        const float in_y = (crop_height_ > 1)
          ? y1 * (image_height_ - 1) + y * height_scale
          : 0.5 * (y1 + y2) * (image_height_ - 1);
        if (in_y < 0 || in_y > image_height_ - 1) {
          for (int x = 0; x < crop_width_; ++x) {
            top_data[(d * crop_height_ + y) * crop_width_ + x] = extrapolation_value_;
          }
          continue;
        }
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width_; ++x) {
          const float in_x = (crop_width_ > 1)
            ? x1 * (image_width_ - 1) + x * width_scale
            : 0.5 * (x1 + x2) * (image_width_ - 1);
          if (in_x < 0 || in_x > image_width_ - 1) {
            top_data[(d * crop_height_ + y) * crop_width_ + x] = extrapolation_value_;
            continue;
          }
          const int left_x_index = floorf(in_x);
          const int right_x_index = ceilf(in_x);
          const float x_lerp = in_x - left_x_index;

          int index = (d * image_height_ + top_y_index) * image_width_ + left_x_index;
          const float top_left = static_cast<float>(image[index]);

          index = (d * image_height_ + top_y_index) * image_width_ + right_x_index;
          const float top_right = static_cast<float>(image[index]);

          index = (d * image_height_ + bottom_y_index) * image_width_ + left_x_index;
          const float bottom_left = static_cast<float>(image[index]);

          index = (d * image_height_ + bottom_y_index) * image_width_ + right_x_index;
          const float bottom_right = static_cast<float>(image[index]);

          const float top = top_left + (top_right - top_left) * x_lerp;
          const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
          top_data[(d * crop_height_ + y) * crop_width_ + x] = top + (bottom - top) * y_lerp;
        }
      }
    }
  }
}


template <typename Dtype>
inline int PyramidROIAlignLayer<Dtype>::get_roi_level(const Dtype *box, const float alpha){
  int k;
  Dtype level;
  level = (box[2] - box[0]) * (box[3] - box[1]);
  level = sqrt(level) / alpha;
  // level = level > 0.0001 ? level : 0.0001;
  level = std::log2(level);
  k = int(std::rint(level) + 4);
  k = k > 2 ? k : 2;
  k = k < 5 ? k : 5;
  // std::cout << k << box[0] << box[1] << box[2] << box[3] << std::endl;
  // exit(0);
  return k;
}


template <typename Dtype>
void PyramidROIAlignLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Dtype *pooled_rois = top[0]->mutable_cpu_data();
  const Dtype *boxes = bottom[0]->cpu_data();
  const Dtype *image_meta = bottom[1]->cpu_data();
  const int batch_size = bottom[0]->shape(0);
  const int num_boxes = bottom[0]->shape(1);
  const int meta_data_size = bottom[1]->shape(-1);
  // bottom[2:end] are feature maps
  // loop over batch

  for(int b = 0; b < batch_size; ++b){
    const int height_ = image_meta[meta_data_size * b + 4];
    const int width_ = image_meta[meta_data_size * b + 5];
    // calculate roi_levels from bottom[0](boxes)
    const int image_area = height_ * width_;
    const float alpha = 224.0 / sqrt(image_area);
    // loop over boxes
    for (int i = 0; i < num_boxes; ++i){
      const int roi_level = get_roi_level(boxes + 4*(b*num_boxes + i), alpha);
      const Dtype *feature_map = const_cast<Dtype*>(bottom[roi_level]->cpu_data()) + bottom[roi_level]->count(1) * b;
      const Dtype *box_current = const_cast<Dtype*>(boxes) + 4*(b*num_boxes + i);
      if(data_format_ == "NHWC"){
        const int image_height_ = bottom[roi_level]->shape(1);
        const int image_width_ = bottom[roi_level]->shape(2);
        const int channels_ = bottom[roi_level]->shape(3);
        // use bilienear algorithm
        crop_and_resize(feature_map, box_current, pooled_rois,
                        image_height_, image_width_, channels_, data_format_);
      }
      // data format, evlayer can use this data format (NCHW) only
      else {
        const int channels_ = bottom[roi_level]->shape(1);
        const int image_height_ = bottom[roi_level]->shape(2);
        const int image_width_ = bottom[roi_level]->shape(3);
        crop_and_resize(feature_map, box_current, pooled_rois,
                        image_height_, image_width_, channels_, data_format_);
      }
      // save next pooled feature
      pooled_rois += top[0]->count(2);
    }
  }
}

INSTANTIATE_CLASS(PyramidROIAlignLayer);
REGISTER_LAYER_CLASS(PyramidROIAlign);

} // namespace caffe
