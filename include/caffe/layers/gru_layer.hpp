#ifndef CAFFE_GRU_LAYER_HPP_
#define CAFFE_GRU_LAYER_HPP_

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
 * @brief Processes sequential inputs using a "Gated Recurrent Unit" (GRU)
 *        [1] style recurrent neural network (RNN). Implemented by unrolling
 *        the GRU computation through time.
 *
 *     Equations (Default: f=Sigmoid, g=Tanh):
 *     X - input tensor
 *     z - update gate
 *     r - reset gate
 *     h - hidden gate
 *     t - time step (t-1 means previous time step)
 *     W[zrh] - W parameter weight matrix for update, reset, and hidden gates

 *     R[zrh] - R recurrence weight matrix for update, reset, and hidden gates

 *     Wb[zrh] - W bias vectors for update, reset, and hidden gates

 *     Rb[zrh] - R bias vectors for update, reset, and hidden gates

 *     WB[zrh] - W parameter weight matrix for backward update, reset, and hidden gates

 *     RB[zrh] - R recurrence weight matrix for backward update, reset, and hidden gates

 *     WBb[zrh] - W bias vectors for backward update, reset, and hidden gates

 *     RBb[zrh] - R bias vectors for backward update, reset, and hidden gates

 *     H - Hidden state

 *     num_directions - 2 if direction == bidirectional else 1

////////////////////////////////////////////////////////////////////////////////////////////////////
// - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)                                                  //
//                                                                                                //
// - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)                                                  //
//                                                                                                //
// - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0 //
//                                                                                                //
// - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0       //
//                                                                                                //
// - Ht = (1 - zt) (.) ht + zt (.) Ht-1                                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////
* Inputs:
 1. X, shape (T, N, input_size)
  - T is the time step
  - N is the number of the independent streams
 2. continue flag, shape (T, N)
 3. X_static (not support)
 4. init_hidden_state, shape (1, N, num_output)

* Outputs:
 1. outputs, shape (T, N, num_output)
 2. final_hidden_state, shape (1, N, num_ouput)

* Shapes of weights and bias:
 1. W: (3*num_ouptut, input_size)
 2. B: (3*num_output,)
 3. R_zr: (2*num_output, num_output)
 4. R_h: (num_output, num_output)
 5. Rb_h: (num_output,)  only available when linear_before_reset is not zero.
    - linear_before_reset is introduced by [[https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU][ONNX GRU]]
*/
template <typename Dtype>
class GRULayer : public RecurrentLayer<Dtype> {
 public:
  explicit GRULayer(const LayerParameter& param)
      : RecurrentLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "GRU"; }

 protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
  virtual void OutputBlobNames(vector<string>* names) const;
};


}  // namespace caffe

#endif  // CAFFE_GRU_LAYER_HPP_
