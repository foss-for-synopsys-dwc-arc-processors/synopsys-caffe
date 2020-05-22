#ifndef CAFFE_MASKRCNN_PROPOSAL_LAYER_HPP_
#define CAFFE_MASKRCNN_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Compute elementwise add. Support broadcasting as numpy and TensorFlow do.
 *
 * Note: Two dimensions are compatible for broadcasting if both are the same or either is 1.
 * The rule starts with the right-most dimension, and works towards the left-most dimension.
 */
  
 

 
 

template <typename Dtype>
class MaskRCNNProposalLayer : public Layer<Dtype> {
 public:
  
  explicit MaskRCNNProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
	  
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
      const vector<Blob<Dtype> *> &top);
	    
	  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);	  
 protected:
 
 // virtual vector<int> argsort(vector<float> &v);
 
  virtual vector<vector<Dtype>> apply_box_deltas_graph(vector<vector<Dtype>> boxes,vector<vector<Dtype>> deltas);
  virtual vector<vector<Dtype>> clip_boxes_graph(vector<vector<Dtype>> boxes,vector<Dtype> window);
  virtual vector<int> image_nms(vector<vector<Dtype>>& bounding_boxes, vector<int>& score, int topk, float threshold);
  virtual Dtype* maskrcnnproposal(const Dtype* rpn_class,const Dtype* rpn_bbox,const Dtype* anchors);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  };
  
  int batch_size;
  int images_per_gpu;
  vector<Dtype> rpn_bbox_std_dev;
  int pre_nms_limit; 
  float rpn_nms_threshold;
  int post_nms_rois_inference;	
  int height;
  int width;
  
  int num_rois;
  
	  


 
  

};

}  // namespace caffe

#endif  // CAFFE_ADD_LAYER_HPP_
