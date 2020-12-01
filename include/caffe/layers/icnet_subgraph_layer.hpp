#ifndef CAFFE_ICNET_SUBGRAPH_LAYER_HPP_
#define CAFFE_ICNET_SUBGRAPH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Merge the AVE Pooling, Interp and Eltwise sum for host_fixed usage in ICNet model.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class ICNetSubgraphLayer : public Layer<Dtype> {
 public:
  explicit ICNetSubgraphLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ICNetSubgraph"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		NOT_IMPLEMENTED;
  }

  int pad_h_ = 0;
  int pad_w_ = 0;
  int num_;
  int channels_;
  int height_, width_;

  vector<int> kernel_h_;
  vector<int> kernel_w_;
  vector<int> stride_h_;
  vector<int> stride_w_;
  vector<int> pooled_height_;
  vector<int> pooled_width_;
  int pool_branches_ = 4;

  Blob<Dtype> pooling_;
  Blob<Dtype> interp_;
};

}  // namespace caffe

#endif  // CAFFE_ICNET_SUBGRAPH_LAYER_HPP_
