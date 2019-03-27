#ifndef CAFFE_WHERE4_GATHERND_CROP_LAYER_HPP_
#define CAFFE_WHERE4_GATHERND_CROP_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"



namespace caffe {

template <typename Dtype>
class Where4GatherndCropLayer : public Layer<Dtype> {
public:
 explicit Where4GatherndCropLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	 const vector<Blob<Dtype>*>& top);
 virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	 const vector<Blob<Dtype>*>& top);

 virtual inline const char* type() const { return "Where4GatherndCrop"; }
 virtual inline int ExactNumBottomBlobs() const { return 6; }
 // bottom[0] is input (params), and bottom[1] provides indices (from where Op).
 // bottom 2~5 provide the input image; to make things easy, assume the box batch_size is 1 for all.
 virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Crop_And_Resize(const Dtype *bottom_data, const Dtype *bottom_rois,
 		Dtype *top_data, int num_boxes_, int image_height_, int image_width_);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	  NOT_IMPLEMENTED;
  }
 //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 //    const vector<Blob<Dtype>*>& top) {}
 //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
 //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

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
};

}  // namespace caffe

#endif  // CAFFE_WHERE4_GATHERND_CROP_LAYER_HPP_

