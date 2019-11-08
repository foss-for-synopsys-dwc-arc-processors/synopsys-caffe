#ifndef CAFFE_ONE_HOT_LAYER_HPP_
#define CAFFE_ONE_HOT_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * Note: implementation of tf.one_hot
 * https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/one_hot
 */

template <typename Dtype> class OneHotLayer : public Layer<Dtype> {
public:
  explicit OneHotLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "OneHot"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
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

  inline vector<int> indices(int num, const vector<int> &shape) const;
  inline int offset(const vector<Blob<Dtype> *> &top,
                    const vector<int> &axis_ind,
                    const vector<int> &indices) const;

  int depth;
  int axis;
  float on_value;
  float off_value;
  int num_axes;
};

} // namespace caffe

#endif // CAFFE_ONE_HOT_LAYER_HPP_
