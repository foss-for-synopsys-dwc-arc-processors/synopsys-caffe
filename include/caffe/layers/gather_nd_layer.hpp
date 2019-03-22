#ifndef CAFFE_GATHER_ND_LAYER_HPP_
#define CAFFE_GATHER_ND_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"



namespace caffe {
/*
* @brief Resize images to size using nearest neighbor interpolation. ////
* Note: implementation of tf.gather_nd
* https://www.tensorflow.org/api_docs/python/tf/gather_nd
*/

template <typename Dtype>
class GatherNdLayer : public Layer<Dtype> {
public:
 explicit GatherNdLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	 const vector<Blob<Dtype>*>& top);
 virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	 const vector<Blob<Dtype>*>& top);

 virtual inline const char* type() const { return "GatherNd"; }
 virtual inline int ExactNumBottomBlobs() const { return 1; }
 virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
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

  int gather_nd_size_;         
  int indices_dim_;
  int indices_N_;
  vector<int> indices_;  
  vector<int> indices_shape_;  
};

}  // namespace caffe

#endif  // CAFFE_GATHER_ND_LAYER_HPP_

