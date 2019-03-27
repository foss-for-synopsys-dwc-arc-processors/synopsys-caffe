#ifndef CAFFE_CROP_AND_RESIZE_LAYERS_HPP_
#define CAFFE_CROP_AND_RESIZE_LAYERS_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{

template <typename Dtype>
class CropAndResizeLayer : public Layer<Dtype>
{
public:
  explicit CropAndResizeLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "CropAndResize"; }

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  //virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  //virtual inline int MaxTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
		  const vector<Blob<Dtype> *> &bottom){
	    NOT_IMPLEMENTED;
	  }

  int batch_size_;
  int channels_;
  int image_height_;
  int image_width_;

  int num_boxes_;
  int crop_height_;
  int crop_width_;

  float extrapolation_value_;
};

} // namespace caffe

#endif // CAFFE_CROP_AND_RESIZE_LAYERS_HPP_
