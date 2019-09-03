#ifndef CAFFE_STRIDED_SLICE_LAYER_HPP_
#define CAFFE_STRIDED_SLICE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype> class StridedSliceLayer : public Layer<Dtype> {
public:
  explicit StridedSliceLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "StridedSlice"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }

  inline vector<int> strided_slice(const vector<int> &t_shape,
                                   const vector<int> &begin,
                                   const vector<int> &strides) const;
  inline vector<int> Indices(int offset, const vector<int> &top_shape,
                             const vector<int> &strided_idx) const;
  inline int offset(const vector<int> &b_shape,
                    const vector<int> &indices) const;

  vector<int> strided_begin_;
  vector<int> strided_end_;
  vector<int> strides_;
  vector<int> t_shape;
  vector<int> t_shape2;
  vector<int> b_shape;
  vector<int> axes_;
  int onnx_flag_;
  int s_len_;
  int begin_mask_;
  int end_mask_;
  int ellipsis_mask_;
  int new_axis_mask_;
  int shrink_axis_mask_;
};

} // namespace caffe

#endif // CAFFE_STRIDED_SLICE_LAYER_HPP_
