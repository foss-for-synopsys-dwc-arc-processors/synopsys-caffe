#ifndef CAFFE_HARD_SWISH_LAYER_HPP_
#define CAFFE_HARD_SWISH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

// the implement of y = x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)
template <typename Dtype> class HardSwishLayer : public NeuronLayer<Dtype> {
public:
  explicit HardSwishLayer(const LayerParameter &param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char *type() const { return "HardSwish"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) {
        NOT_IMPLEMENTED;
      }
    }
  }
};

} // namespace caffe

#endif // CAFFE_HARD_SWISH_LAYER_HPP_
