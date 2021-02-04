#ifndef CAFFE_PROPOSAL_LAYER_HPP_
#define CAFFE_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))
namespace caffe {

/**
 * @brief Provides ROIs by assigning tops directly.
 * should generate same results as the python version
 * Ref: https://github.com/foss-for-synopsys-dwc-arc-processors/py-faster-rcnn
 */
template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
 public:
  explicit ProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Proposal"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  //virtual inline int MinBottomBlobs() const { return 3; }
  //virtual inline int MaxBottomBlobs() const { return 3; }
  //virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; } // output rois
  virtual inline int MaxTopBlobs() const { return 2; } // optional, output scores

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  virtual void Generate_anchors();

  virtual void _whctrs(vector <float> anchor, vector<float> &ctrs);

  virtual void _ratio_enum(vector <float> anchor, vector <float> &anchors_ratio);

  virtual void _mkanchors(vector<float> ctrs, vector<float> &anchors);

  virtual void _scale_enum(vector<float> anchors_ratio, vector<float> &anchor_boxes);

  virtual void bbox_transform_inv(int img_width, int img_height, vector<vector<float> > bbox, vector<vector<float> > select_anchor, vector<vector<float> > &pred);

  //virtual void apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence);

  virtual void filter_boxes(vector<vector<float> > &pred_boxes, vector<float> &confidence, float min_size);
  virtual void applynmsfast(vector<vector<float> > &pred_boxes,vector<pair<Dtype, int> > &score_index_vec,
      const float nms_threshold, const int top_k, vector<int> &indices);

  int feat_stride_; //resolution
  int anchor_base_size_;
  vector<float> anchor_scale_; //anchor scale
  vector<float> anchor_ratio_; //anchor_ratio

  int max_rois_;
  vector<float> anchor_boxes_;
  int pre_nms_topn_;
  float rpn_min_size_;
  float rpn_nms_thresh_;

};

}  // namespace caffe

#endif  // CAFFE_PROPOSAL_LAYER_HPP_
