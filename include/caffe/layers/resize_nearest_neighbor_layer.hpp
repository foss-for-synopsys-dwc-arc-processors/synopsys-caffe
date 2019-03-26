#ifndef CAFFE_RESIZENEARESTNEIGHBOR_LAYER_HPP_
#define CAFFE_RESIZENEARESTNEIGHBOR_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Resize images to size using nearest neighbor interpolation.
 *
 * Note: implementation of tf.resize_nearest_neighbor
 * https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest_neighbor
 */
template <typename Dtype>
class ResizeNearestNeighborLayer : public Layer<Dtype> {
 public:
  explicit ResizeNearestNeighborLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ResizeNearestNeighbor"; }
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
  int output_height;
  int output_width;
  bool align_corners;
  string data_format;
};

}  // namespace caffe

#endif  // CAFFE_RESIZENEARESTNEIGHBOR_LAYER_HPP_
