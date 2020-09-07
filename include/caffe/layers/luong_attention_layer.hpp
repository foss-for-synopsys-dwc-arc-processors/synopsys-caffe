#ifndef CAFFE_LUONG_ATTENTION_LAYER_HPP_
#define CAFFE_LUONG_ATTENTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LuongAttentionLayer : public Layer<Dtype> {
 public:
  
  explicit LuongAttentionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
      const vector<Blob<Dtype> *> &top);
	    
	  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);	  
	  

	  
  virtual inline const char *type() const { return "LuongAttention"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }	  
	  
	  
	  
  protected:

  inline void softmax(Dtype* score);
  inline Dtype* calculate_attention(const Dtype *query,const Dtype *state);
  inline Dtype* transpose(const Dtype *memory);
  inline Dtype* luong_score(const Dtype* query,Dtype* _memory);
  inline Dtype* matmul(const Dtype* query,Dtype* keys);
  
 

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
	
   };
  
  const Dtype* keys;
  int batch_size;        
  int query_depth;
  int alignments_size;

};

}  // namespace caffe

#endif  // CAFFE_ADD_LAYER_HPP_
