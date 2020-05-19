#ifndef CAFFE_PYRAMID_ROI_ALIGN_LAYER_HPP_
#define CAFFE_PYRAMID_ROI_ALIGN_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

/*
  """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
*/

namespace caffe {

template <typename Dtype> class PyramidROIAlignLayer : public Layer<Dtype> {
public:
  explicit PyramidROIAlignLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "PyramidROIAlign"; }
  virtual inline int ExactNumBottomBlobs() const { return 6; }
  // bottom[0] is input (params), and bottom[1] provides indices (from where
  // Op). bottom 2~5 provide the input image; to make things easy, assume the
  // box batch_size is 1 for all.
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void crop_and_resize(const Dtype *image, const Dtype *box, Dtype *top_data,
                               const int image_height_, const int image_width_,
                               const int channels_, const string data_format_);

  virtual int get_roi_level(const Dtype *box, const float alpha);

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

  int crop_height_;
  int crop_width_;
  float extrapolation_value_;
  string data_format_;
};

} // namespace caffe

#endif // CAFFE_PYRAMID_ROI_ALIGN_LAYER_HPP_
