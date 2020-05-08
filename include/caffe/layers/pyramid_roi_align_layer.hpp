#ifndef CAFFE_PYRAMID_ROI_ALIGN_LAYER_HPP_
#define CAFFE_PYRAMID_ROI_ALIGN_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

/*
Combine the functionality of where4 + GatherNd + CropAndResize with
where4 + Topk + Gather layers to work around the dynamic problem in
Mask RCNN implementation.
*/

namespace caffe {

template <typename Dtype> class PyramidRoiAlignLayer : public Layer<Dtype> {
public:
  explicit PyramidRoiAlignLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "PyramidRoiAlign"; }
  virtual inline int ExactNumBottomBlobs() const { return 6; }
  // bottom[0] is input (params), and bottom[1] provides indices (from where
  // Op). bottom 2~5 provide the input image; to make things easy, assume the
  // box batch_size is 1 for all.
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Crop_And_Resize(const Dtype *bottom_data,
                               const Dtype *bottom_rois, Dtype *top_data,
                               int num_boxes_, int image_height_,
                               int image_width_, string data_format_);

  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top) {}
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  //    {}

  size_t num_output_;
  int axis_;

  int gather_nd_size_;
  int indices_dim_;
  int indices_N_;
  vector<int> indices_;
  vector<int> indices_shape_;
  Blob<Dtype> gather_output_;

  int channels_;
  int crop_height_;
  int crop_width_;
  float extrapolation_value_;
  Blob<Dtype> crop_output_;

  vector<int> topk_indices_;

  int num_gather_;
  int gather_size_;
  int gather_axis_;
  int indices_dim_out_;
  vector<int> indices_shape_out_;

  string data_format_;
};

} // namespace caffe

#endif // CAFFE_PYRAMID_ROI_ALIGN_LAYER_HPP_
