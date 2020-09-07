#ifndef CAFFE_ATTENTION_LAYER_HPP_
#define CAFFE_ATTENTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class AttentionLayer : public Layer<Dtype> {
	public:

		explicit AttentionLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top);
		
		
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);	  
		

		
		virtual inline const char *type() const { return "Attention"; }
		virtual inline int ExactNumBottomBlobs() const { return 4; }
		virtual inline int ExactNumTopBlobs() const { return 1; }	  
		
		
		
	protected:

		inline void softmax(Dtype* score);
		inline Dtype* calculate_attention(const Dtype *query,const Dtype *state);
		inline Dtype* transpose(const Dtype *memory);
		inline Dtype* luong_score(const Dtype* query,Dtype* _memory);
		inline Dtype* matmul(const Dtype* query,Dtype* keys);

		inline Dtype* bahdanau_score(const Dtype* query,Dtype* memory, Dtype* attention_v, Dtype attention_g, Dtype* attention_b);
		





		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
			NOT_IMPLEMENTED;
			
		};
		
		const Dtype* keys;
		const Dtype* attention_v;
		int batch_size = 2;
		int query_depth = 1024;
		int alignments_size = 500;
		int scale_weight = NULL;
		string probability_fn = "softmax";
		Dtype* attention_b = NULL;
		//   Dtype* attention_v = NULL;
		Dtype attention_g = NULL;
		bool scale = false;
		bool normalize = false; 
		string attention_option = "normed_bahdanau";
		
		
		  
	};

}  // namespace caffe

#endif  // CAFFE_ADD_LAYER_HPP_
