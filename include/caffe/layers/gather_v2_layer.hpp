#ifndef CAFFE_GATHER_V2_LAYER_HPP_
#define CAFFE_GATHER_V2_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * @brief Resize images to size using nearest neighbor interpolation. ////
 * Note: another implementation of tf.gather
 * https://www.tensorflow.org/api_docs/python/tf/gather
 * In GatherV2, params and indices are inputs, axis is attribute
 */

template <typename Dtype> class GatherV2Layer : public Layer<Dtype> {
public:
  explicit GatherV2Layer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "GatherV2"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
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

  int gather_axis_;
  int indices_dim_;
  vector<int> indices_shape_;
};

} // namespace caffe

#endif // CAFFE_GATHER_V2_LAYER_HPP_
