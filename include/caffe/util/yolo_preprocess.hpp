#ifndef YOLO_PREPROCESS_HPP
#define YOLO_PREPROCESS_HPP

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

float rand_uniform(float min, float max);

float rand_scale(float value_lower, float value_upper);

void set_pixel(float *m, int w, int h, int ch, int x, int y, int c, float val);

void set_pixel_with_scaling(float *m, int w, int h, int ch, int x, int y, int c,
                            float val, float scale);

float get_pixel(float *m, int w, int h, int ch, int x, int y, int c);

void scale_image_channel(float *im, int w, int h, int ch, int c, float v);

float three_way_max(float a, float b, float c);

float three_way_min(float a, float b, float c);

void rgb_to_hsv(float *im, int width, int height, int channels);

void hsv_to_rgb(float *im, int width, int height, int channels);

void constrain_image(float *im, int w, int h, int c);

float constrain(float min, float max, float a);

void distort_image(float *im, int w, int h, int c, float hue, float sat,
                   float val);

void random_distort_image(float *im, int w, int h, int c, float hue,
                          float saturation_lower, float saturation_upper,
                          float exposure_lower, float exposure_upper);

void flip_image(float *im, int w, int h, int c);

#ifdef USE_OPENCV
void bgr_to_rgb(cv::Mat im);

cv::Mat hwc_to_chw(cv::Mat im);

float get_pixel_image(const cv::Mat& m, int x, int y, int c);

float get_pixel_extend(const cv::Mat& m, int x, int y, int c);

float bilinear_interpolate(const cv::Mat& im, float x, float y, int c);

void place_image(cv::Mat im, int w, int h, int dx, int dy, float *resized_image,
                 int resize_w, int resize_h, float scale);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // YOLO_PREPROCESS_HPP
