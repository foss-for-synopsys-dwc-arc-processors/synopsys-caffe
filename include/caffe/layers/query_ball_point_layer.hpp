#ifndef CAFFE_QUERY_BALL_POINT_LAYER_HPP_
#define CAFFE_QUERY_BALL_POINT_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class QueryBallPointLayer : public Layer<Dtype> {
 public:
  explicit QueryBallPointLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QueryBallPoint"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int n_sample_point_; // the number of (farthest) point to sample
  // so far, we leave the random first node un-implemented, but choose the index-0 point
  float radius_;
  //int B, N, D; // batch_size, num_points, Dimension (3d, xyz)
};

}  // namespace caffe

#endif  // CAFFE_QUERY_BALL_POINT_LAYER_HPP_
