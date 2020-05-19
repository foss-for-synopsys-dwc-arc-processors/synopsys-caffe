#ifndef CAFFE_MASKRCNN_DETECTION_LAYER_HPP_
#define CAFFE_MASKRCNN_DETECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// implement of
// https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py#L782

template <typename Dtype> class MaskRCNNDetectionLayer : public Layer<Dtype> {
public:
  explicit MaskRCNNDetectionLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "MaskRCNNDetection"; }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  inline vector<int> tf_nms(vector<vector<Dtype>> &pred_boxes,
                            vector<int> &order, int topk,
                            float iou_threshold) const;

  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  //    {
  //  NOT_IMPLEMENTED;
  //}

  // maskrcnn_detection params
  int batch_size_;
  int detection_max_instances_;
  float detection_min_confidence_;
  float detection_nms_threshold_;
  vector<float> bbox_std_dev_;

  int N_;
  int num_class_;
};

} // namespace caffe

#endif // CAFFE_MASKRCNN_DETECTION_LAYER_HPP_
