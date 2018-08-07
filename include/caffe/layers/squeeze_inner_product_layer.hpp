/***************************** MulticoreWare_Modified - Feature: Pruning / Splicing ************************************/
#ifndef CAFFE_SQUEEZE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_SQUEEZE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SqueezeInnerProductLayer : public Layer<Dtype> {
 public:
  explicit SqueezeInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SqueezeInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual void AggregateParams(const int, const Dtype* , const Dtype* ,
    unsigned int* );
  virtual void CalculateMask(const int, const Dtype* , Dtype* ,
    Dtype , Dtype , Dtype );

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

 private:
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

#endif  // CAFFE_SQUEEZE_INNER_PRODUCT_LAYER_HPP_
/***********************************************************************************************************************/
