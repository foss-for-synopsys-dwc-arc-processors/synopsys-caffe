#ifndef CAFFE_PEEPHOLE_LSTM_LAYER_HPP_
#define CAFFE_PEEPHOLE_LSTM_LAYER_HPP_

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
 * o - output gate
 *
 * f - forget gate
 *
 * c - cell gate
 *
 * t - time step (t-1 means previous time step)
 *
 * W[iofc] - W parameter weight matrix for input, output, forget, and cell gates
 *
 * R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates
 *
 * Wb[iofc] - W bias vectors for input, output, forget, and cell gates
 *
 * Rb[iofc] - R bias vectors for input, output, forget, and cell gates
 *
 * P[iof] - P peephole weight vector for input, output, and forget gates
 *
 * WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates
 *
 * RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates
 *
 * WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates
 *
 * RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates
 *
 * PB[iof] - P peephole weight vector for backward input, output, and forget gates
 *
 * H - Hidden state

 * num_directions - 2 if direction == bidirectional else 1

       /////////////////////////////////////////////////////////////////
       // - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi) //
       //                                                             //
       // - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf) //
       //                                                             //
       // - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)               //
       //                                                             //
       // - Ct = ft (.) Ct-1 + it (.) ct                              //
       //                                                             //
       // - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)   //
       //                                                             //
       // - Ht = ot (.) h(Ct)                                         //
       /////////////////////////////////////////////////////////////////
 * ONNX specification end

 * Inputs:
  1. X, shape (T, N, input_size)
   - T is the time step
   - N is the number of the independent streams
  2. continue flag, shape (T, N)
  3. X_static (optional, (N, input_size))
  4. init_hidden_state, shape (1, N, num_output)
  5. init_cell_state, shape (1, N, num_output)

 * Outputs:
  1. outputs, shape (T, N, num_output)
  2. final_hidden_state, shape (1, N, num_ouput)
  3. final_cell_state, shape (1, N, num_output)

 * Shapes of weights and bias:
  1. W: (4*num_ouptut, input_size)
  2. B: (4*num_output,)
  3. W_static (optional, (4*num_output, input_size))
  4. R: (4*num_output, num_output)
  5. Pi: (num_output)
  6. Pf: (num_output)
  7. Po: (num_output)
*/
template <typename Dtype>
class PeepholeLSTMLayer : public RecurrentLayer<Dtype> {
 public:
  explicit PeepholeLSTMLayer(const LayerParameter& param)
      : RecurrentLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "PeepholeLSTM"; }

 protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
  virtual void OutputBlobNames(vector<string>* names) const;
};


}  // namespace caffe

#endif  // CAFFE_PEEPHOLE_LSTM_LAYER_HPP_
