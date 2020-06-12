#ifndef CAFFE_SIMPLE_RNN_LAYER_HPP_
#define CAFFE_SIMPLE_RNN_LAYER_HPP_

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

* ONNX specification
* Notations:
* X - input tensor
*
* i - input gate
*
* t - time step (t-1 means previous time step)
*
* W[i] - W parameter weight matrix for input gates
*
* R[i] - R recurrence weight matrix for input gates
*
* Wb[i] - W bias vectors for input gates
*
* Rb[i] - R bias vectors for input gates
*
* WB[i] - W parameter weight matrix for backward input, output, forget, and cell gates
*
* RB[i] - R recurrence weight matrix for backward input, output, forget, and cell gates
*
* WBb[i] - W bias vectors for backward input, output, forget, and cell gates
*
* RBb[i] - R bias vectors for backward input, output, forget, and cell gates
*
* H - Hidden state
* num_directions - 2 if direction == bidirectional else 1
/////////////////////////////////////////////////////////////////
// - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)               //
/////////////////////////////////////////////////////////////////
* ONNX specification end
* Inputs:
1. X, shape (T, N, input_size)
- T is the time step
- N is the number of the independent streams
2. continue flag, shape (T, N)
3. X_static (optional, (N, input_size))
4. init_hidden_state, shape (1, N, num_output)

* Outputs:
1. outputs, shape (T, N, num_output)
2. final_hidden_state, shape (1, N, num_ouput)
* Shapes of weights and bias:
1. W: (num_ouptut, input_size)
2. B: (num_output,)
3. W_static (optional, (num_output, input_size))
4. R: (num_output, num_output)
 */
template <typename Dtype>
class SimpleRNNLayer : public RecurrentLayer<Dtype> {
 public:
  explicit SimpleRNNLayer(const LayerParameter& param)
      : RecurrentLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "SimpleRNN"; }

 protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
  virtual void OutputBlobNames(vector<string>* names) const;
};

}  // namespace caffe

#endif  // CAFFE_SIMPLE_RNN_LAYER_HPP_

