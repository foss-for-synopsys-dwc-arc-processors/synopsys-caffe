#ifndef CAFFE_LSTM_V2_LAYER_HPP_
#define CAFFE_LSTM_V2_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype> class RecurrentLayer;

/**
 * @brief Processes sequential inputs using a "Long Short-Term Memory" (LSTM)
 *        [1] style recurrent neural network (RNN). Implemented by unrolling
 *        the LSTM computation through time.
 *
 * The specific architecture used in this implementation is as described in
 * "Learning to Execute" [2], reproduced below:
 * f_t &:= \mathcal{F}(W_f x_t + R_f h_{t-1} + B_f)
 * o_t &:= \mathcal{F}(W_o x_t + R_o h_{t-1} + B_o)
 * g_t &:= \mathcal{G}(W_g x_t + R_g h_{t-1} + B_g)
 * c_t &:= (f_t \odot c_{t-1}) + (i_t \odot g_t)
 * h_t &:= o_t \odot \mathcal{H}(c_t)
 * In the implementation, the i, f, o, and g computations are performed as a
 * single inner product.
 *
 * Notably, this implementation lacks the "diagonal" gates, as used in the
 * LSTM architectures described by Alex Graves [3] and others.
 *
 * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
 *     Neural Computation 9, no. 8 (1997): 1735-1780.
 *
 * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
 *     arXiv preprint arXiv:1410.4615 (2014).
 *
 * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
 *     arXiv preprint arXiv:1308.0850 (2013).
 */
template <typename Dtype>
class LSTMV2Layer : public RecurrentLayer<Dtype> {
 public:
  explicit LSTMV2Layer(const LayerParameter& param)
      : RecurrentLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "LSTMV2"; }

 protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
  virtual void OutputBlobNames(vector<string>* names) const;
};

}  // namespace caffe

#endif  // CAFFE_LSTM_V2_LAYER_HPP_
