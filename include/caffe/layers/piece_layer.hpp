#ifndef CAFFE_PIECE_LAYER_HPP_
#define CAFFE_PIECE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype> class PieceLayer : public Layer<Dtype> {
public:
  explicit PieceLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Piece"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }

  inline vector<int> piece(const vector<int> &begin,
                           const vector<int> &size) const;
  inline vector<int> Indices(int offset, const vector<int> &top_shape,
                             const vector<int> &piece_idx) const;
  inline int offset(const vector<Blob<Dtype> *> &bottom,
                    const vector<int> &indices) const;
  inline int bottom_count(const vector<int> &shape, const int axis) const;

  vector<int> piece_begin_;
  vector<int> piece_size_;
};

} // namespace caffe

#endif // CAFFE_PIECE_LAYER_HPP_
