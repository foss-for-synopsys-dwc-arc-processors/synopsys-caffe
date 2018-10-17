#include "caffe/util/yolo_preprocess.hpp"

namespace caffe {

float rand_uniform(float min, float max) {
  if(max < min){
      float swap = min;
      min = max;
      max = swap;
  }
  return ((float)rand()/RAND_MAX * (max - min)) + min;
}

float rand_scale(float value_lower, float value_upper) {
  float scale = rand_uniform(value_lower, value_upper);
  if(rand() % 2) return scale;
  return 1./scale;
}

void set_pixel(float *m, int w, int h, int ch, int x, int y, int c, float val) {
  if (x < 0 || y < 0 || c < 0 || x >= w || y >= h || c >= ch) return;
  assert(x < w && y < h && c < ch);
  m[c*h*w + y*w + x] = val;
}

void set_pixel_with_scaling(float *m, int w, int h, int ch, int x, int y, int c,
                            float val, float scale) {
  if (x < 0 || y < 0 || c < 0 || x >= w || y >= h || c >= ch) return;
  assert(x < w && y < h && c < ch);
  m[c*h*w + y*w + x] = val * scale;
}

float get_pixel(float *m, int w, int h, int ch, int x, int y, int c) {
  assert(x < w && y < h && c < ch);
  return m[c*h*w + y*w + x];
}

void scale_image_channel(float *im, int w, int h, int ch, int c, float v) {
  int i, j;
  for(j = 0; j < h; ++j){
      for(i = 0; i < w; ++i){
          float pix = get_pixel(im, w, h, ch, i, j, c);
          pix = pix*v;
          set_pixel(im, w, h, ch, i, j, c, pix);
      }
  }
}

float three_way_max(float a, float b, float c) {
  return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c);
}

float three_way_min(float a, float b, float c) {
  return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c);
}

void rgb_to_hsv(float *im, int width, int height, int channels) {
  assert(channels == 3);
  int i, j;
  float r, g, b;
  float h, s, v;
  for(j = 0; j < height; ++j){
      for(i = 0; i < width; ++i){
          r = get_pixel(im, width, height, channels, i, j, 0);
          g = get_pixel(im, width, height, channels, i, j, 1);
          b = get_pixel(im, width, height, channels, i, j, 2);
          float max = three_way_max(r,g,b);
          float min = three_way_min(r,g,b);
          float delta = max - min;
          v = max;
          if(max == 0){
              s = 0;
              h = 0;
          }else{
              s = delta/max;
              if(r == max){
                  h = (g - b) / delta;
              } else if (g == max) {
                  h = 2 + (b - r) / delta;
              } else {
                  h = 4 + (r - g) / delta;
              }
              if (h < 0) h += 6;
              h = h/6.;
          }
          set_pixel(im, width, height, channels, i, j, 0, h);
          set_pixel(im, width, height, channels, i, j, 1, s);
          set_pixel(im, width, height, channels, i, j, 2, v);
      }
  }
}

void hsv_to_rgb(float *im, int width, int height, int channels) {
  assert(channels == 3);
  int i, j;
  float r, g, b;
  float h, s, v;
  float f, p, q, t;
  for(j = 0; j < height; ++j){
      for(i = 0; i < width; ++i){
          h = 6 * get_pixel(im, width, height, channels, i, j, 0);
          s = get_pixel(im, width, height, channels, i, j, 1);
          v = get_pixel(im, width, height, channels, i, j, 2);
          if (s == 0) {
              r = g = b = v;
          } else {
              int index = floor(h);
              f = h - index;
              p = v*(1-s);
              q = v*(1-s*f);
              t = v*(1-s*(1-f));
              if(index == 0){
                  r = v; g = t; b = p;
              } else if(index == 1){
                  r = q; g = v; b = p;
              } else if(index == 2){
                  r = p; g = v; b = t;
              } else if(index == 3){
                  r = p; g = q; b = v;
              } else if(index == 4){
                  r = t; g = p; b = v;
              } else {
                  r = v; g = p; b = q;
              }
          }
          set_pixel(im, width, height, channels, i, j, 0, r);
          set_pixel(im, width, height, channels, i, j, 1, g);
          set_pixel(im, width, height, channels, i, j, 2, b);
      }
  }
}

void constrain_image(float *im, int w, int h, int c) {
  int i;
  for(i = 0; i < w*h*c; ++i){
      if(im[i] < 0) im[i] = 0;
      if(im[i] > 1) im[i] = 1;
  }
}

float constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

void distort_image(float *im, int w, int h, int c, float hue, float sat,
                   float val) {
  rgb_to_hsv(im, w, h, c);
  scale_image_channel(im, w, h, c, 1, sat);
  scale_image_channel(im, w, h, c, 2, val);
  int i;
  for(i = 0; i < w*h; ++i){
      im[i] = im[i] + hue;
      if (im[i] > 1) im[i] -= 1;
      if (im[i] < 0) im[i] += 1;
  }
  hsv_to_rgb(im, w, h, c);
  constrain_image(im, w, h, c);
}

void random_distort_image(float *im, int w, int h, int c, float hue,
                          float saturation_lower, float saturation_upper,
                          float exposure_lower, float exposure_upper) {
  float dhue = rand_uniform(-hue, hue);
  float dsat = rand_scale(saturation_lower, saturation_upper);
  float dexp = rand_scale(exposure_lower, exposure_upper);
  distort_image(im, w, h, c, dhue, dsat, dexp);
}

void flip_image(float *im, int w, int h, int c) {
  int i,j,k;
  for(k = 0; k < c; ++k){
      for(i = 0; i < h; ++i){
          for(j = 0; j < w/2; ++j){
              int index = j + w*(i + h*(k));
              int flip = (w - j - 1) + w*(i + h*(k));
              float swap = im[flip];
              im[flip] = im[index];
              im[index] = swap;
          }
      }
  }
}

#ifdef USE_OPENCV

void bgr_to_rgb(cv::Mat im) {
  for(int i = 0; i < im.cols*im.rows; ++i){
      float swap = im.data[i];
      im.data[i] = im.data[i+im.cols*im.rows*2];
      im.data[i+im.cols*im.rows*2] = swap;
  }
}

cv::Mat hwc_to_chw(cv::Mat im) {
  cv::Mat cv_img_temp(im.rows, im.cols, CV_8UC3);
  for(int i = 0; i < im.rows; i++){
      for(int j = 0; j < im.cols; j++){
          for(int k = 0; k < im.channels(); k++) {
              cv_img_temp.data[k*im.cols*im.rows + i*im.cols + j] =
              im.data[i*im.cols*im.channels() + j*im.channels() + k];
          }
      }
  }
  return cv_img_temp;
}

float get_pixel_image(const cv::Mat& m, int x, int y, int c) {
  assert(x < m.cols && y < m.rows && c < m.channels());
  return (float)m.data[c*m.rows*m.cols + y*m.cols + x];
}

float get_pixel_extend(const cv::Mat& m, int x, int y, int c) {
  if(x < 0 || x >= m.cols || y < 0 || y >= m.rows) return 0;
  if(c < 0 || c >= m.channels()) return 0;
  return get_pixel_image(m, x, y, c);
}

float bilinear_interpolate(const cv::Mat& im, float x, float y, int c) {
  int ix = (int) floorf(x);
  int iy = (int) floorf(y);

  float dx = x - ix;
  float dy = y - iy;

  float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) +
              dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) +
              (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
              dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
  return val;
}

void place_image(cv::Mat im, int w, int h, int dx, int dy, float *resized_image,
                 int resize_w, int resize_h, float scale) {
  int x, y, c;
  for(c = 0; c < im.channels(); ++c){
      for(y = 0; y < h; ++y){
          for(x = 0; x < w; ++x){
              float rx = ((float)x / w) * im.cols;
              float ry = ((float)y / h) * im.rows;
              float val = bilinear_interpolate(im, rx, ry, c);
              set_pixel_with_scaling(resized_image, resize_w, resize_h,
                              im.channels(), x + dx, y + dy, c, val, scale);
          }
      }
  }
}
#endif  // USE_OPENCV

}  // namespace caffe
