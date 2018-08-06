/***************************** MulticoreWare_Modified - Feature: Pruning / Splicing ************************************/
#ifndef CAFFE__SQUEEZE_DECONV_LAYER_HPP_
#define CAFFE__SQUEEZE_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ConvolutionLayer.
 *
 *   ConvolutionLayer computes each output value by dotting an input window with
 *   a filter; DeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
 *   reversed. DeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Dtype>
class SqueezeDeconvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit SqueezeDeconvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "SqueezeDeconvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void AggregateParams(const int, const Dtype* , const Dtype* ,
    unsigned int* );
  virtual void CalculateMask(const int, const Dtype* , Dtype* ,
    Dtype , Dtype , Dtype );

  public:
    Blob<Dtype> weight_tmp_;
    Blob<Dtype> bias_tmp_;
    Blob<Dtype> rand_weight_m_;
    Blob<Dtype> rand_bias_m_;
    Dtype gamma,power;
    Dtype crate;
    Dtype mu,std;
    int iter_stop_;
    bool dynamicsplicing;
    float splicing_rate;
};

}  // namespace caffe

#endif  //CAFFE__SQUEEZE_DECONV_LAYER_HPP_
/***********************************************************************************************************************/
