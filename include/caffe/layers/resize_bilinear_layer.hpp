#ifndef CAFFE_RESIZE_BILINEAR_LAYER_HPP_
#define CAFFE_RESIZE_BILINEAR_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Resize images to size using bilinear interpolation.
 *
 * Note: implementation of tf.resize_bilinear
 *
 */
template <typename Dtype>
class ResizeBilinearLayer : public Layer<Dtype> {
 public:
  explicit ResizeBilinearLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ResizeBilinear"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top) {}
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  int output_height_;
  int output_width_;
  bool align_corners_;
  string data_format_;
  bool half_pixel_centers_;
  bool pytorch_half_pixel_;

  // Compute the interpolation indices only once.
  struct CachedInterpolation {
    int lower;  // Lower source index used in the interpolation
    int upper;  // Upper source index used in the interpolation
    // 1-D linear iterpolation scale (see:
    // https://en.wikipedia.org/wiki/Bilinear_interpolation)
    float lerp;
  };

  virtual void compute_interpolation_weights(const int out_size,
      const int in_size,
      const float scale,
      struct CachedInterpolation* interpolation);

};

}  // namespace caffe

#endif  // CAFFE_RESIZE_BILINEAR_LAYER_HPP_
