#ifndef CAFFE_RNN_V2_LAYER_HPP_
#define CAFFE_RNN_V2_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype> class RNNv2Layer : public Layer<Dtype> {
public:
  explicit RNNv2Layer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);
  virtual void Reset();

  virtual inline const char *type() const { return "RNNv2"; }
  virtual inline int MinBottomBlobs() const {
    int min_bottoms = 2;
    vector<string> inputs;
    this->RecurrentBlobNamePrefix(&inputs);
    min_bottoms += inputs.size();
    return min_bottoms;
  }
  virtual inline int MaxBottomBlobs() const { return MinBottomBlobs() + 1; }
  virtual inline int ExactNumTopBlobs() const {
    int num_tops = 1;
    vector<string> outputs;
    this->RecurrentBlobNamePrefix(&outputs);
    num_tops += outputs.size();
    return num_tops;
  }

protected:
  /**
   * @brief Fills net_param with the recurrent network architecture.  Subclasses
   *        should define this -- see RNNLayer for examples.
   */
  void FillUnrolledNet(NetParameter *net_param,
                       const string x_name,
                       const string cont_name,
                       vector<string> output_names,
                       vector<string> recur_name_prefix,
                       const string &layer_name_prefix);

  /**
   * @brief Fills names with the names of the 0th timestep recurrent input
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  void RecurrentBlobNamePrefix(vector<string> *names) const;

  /**
   * @brief Fills shapes with the shapes of the recurrent input Blob&s.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  void RecurrentInputShapes(vector<BlobShape> *shapes) const;

  /**
   * @brief Fills names with the names of the output blobs, concatenated across
   *        all timesteps.  Should return a name for each top Blob.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer for
   *        examples.
   */
  void OutputBlobNames(vector<string> *names) const;

  /**
   * @param bottom input Blob vector (length 2-3)
   *
   *   -# @f$ (T \times N \times ...) @f$
   *      the time-varying input @f$ x @f$.  After the first two axes, whose
   *      dimensions must correspond to the number of timesteps @f$ T @f$ and
   *      the number of independent streams @f$ N @f$, respectively, its
   *      dimensions may be arbitrary.  Note that the ordering of dimensions --
   *      @f$ (T \times N \times ...) @f$, rather than
   *      @f$ (N \times T \times ...) @f$ -- means that the @f$ N @f$
   *      independent input streams must be "interleaved".
   *
   *   -# @f$ (T \times N) @f$
   *      the sequence continuation indicators @f$ \delta @f$.
   *      These inputs should be binary (0 or 1) indicators, where
   *      @f$ \delta_{t,n} = 0 @f$ means that timestep @f$ t @f$ of stream
   *      @f$ n @f$ is the beginning of a new sequence, and hence the previous
   *      hidden state @f$ h_{t-1} @f$ is multiplied by @f$ \delta_t = 0 @f$
   *      and has no effect on the cell's output at timestep @f$ t @f$, and
   *      a value of @f$ \delta_{t,n} = 1 @f$ means that timestep @f$ t @f$ of
   *      stream @f$ n @f$ is a continuation from the previous timestep
   *      @f$ t-1 @f$, and the previous hidden state @f$ h_{t-1} @f$ affects the
   *      updated hidden state and output.
   *
   * @param top output Blob vector (length 1)
   *   -# @f$ (T \times N \times D) @f$
   *      the time-varying output @f$ y @f$, where @f$ D @f$ is
   *      <code>rnn_v2_param.hidden_size()</code>.
   *      Refer to documentation for particular RNNv2Layer implementations
   *      (such as RNNLayer and LSTMLayer) for the definition of @f$ y @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }

  /// @brief A Net to implement the Recurrent functionality.
  shared_ptr<Net<Dtype> > unrolled_net_;

  /// @brief The number of independent streams to process simultaneously.
  int N_;

  /**
   * @brief The number of timesteps in the layer's input, and the number of
   *        timesteps over which to backpropagate through time.
   */
  int T_;

  vector<Blob<Dtype> *> recur_input_blobs_;
  vector<Blob<Dtype> *> recur_output_blobs_;
  vector<Blob<Dtype> *> output_blobs_;
  Blob<Dtype> *x_input_blob_;
  Blob<Dtype> *cont_input_blob_;

  vector<string> activations_;
  vector<float> activation_alpha_;
  vector<float> activation_beta_;
  string direction_;
};

} // namespace caffe

#endif // CAFFE_RNN_V2_LAYER_HPP_
