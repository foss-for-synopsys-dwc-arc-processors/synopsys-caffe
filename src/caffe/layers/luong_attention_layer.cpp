#include <algorithm>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>


#include "caffe/layers/luong_attention_layer.hpp"
using namespace std;
namespace caffe {    

template <typename Dtype>
void LuongAttentionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top){
		
	cout<<"test 8:layer_setup"<<endl;

	keys = NULL;
	
	/*
    batch_size = this->layer_param_.luong_attention_param().batch_size();        
    query_depth = this->layer_param_.luong_attention_param().query_depth();
    alignments_size = this->layer_param_.luong_attention_param().alignments_size();
	*/
	
	batch_size = 2;     
    query_depth = 1024;
    alignments_size = 500;
	
}

template <typename Dtype>
void LuongAttentionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = {batch_size,alignments_size};
  top[0]->Reshape(top_shape);
}


template <typename Dtype>
void LuongAttentionLayer<Dtype>::softmax(Dtype* score){
	
	
	
	for(int b=0;b<batch_size;b++){
		
		Dtype max = *std::max_element(score+b*alignments_size, score+b*alignments_size+alignments_size);
		Dtype sum = 0.0;
		
		
		
		cout<<"max="<<max<<endl;
		
		
		for (int i = 0; i < alignments_size; i++) {
		    score[b*alignments_size+i] = (score[b*alignments_size+i]-max);
			score[b*alignments_size+i] = (Dtype)exp(score[b*alignments_size+i]);
			sum += score[b*alignments_size+i];
		}
			cout<<"sum="<<sum<<endl;
		for (int i = 0; i < alignments_size; i++){

			cout<<"score_after="<<(Dtype)(score[b*alignments_size+i]/(Dtype)sum)<<endl;
			score[b*alignments_size+i] = (Dtype) (score[b*alignments_size+i]/(Dtype)sum);
		}	
	}
}

template <typename Dtype>
Dtype* LuongAttentionLayer<Dtype>::matmul(const Dtype* query,Dtype* keys){
	
	Dtype *score =
      (Dtype *)malloc(batch_size * alignments_size * sizeof(Dtype));
	
	for(int b=0; b<batch_size;b++){
		for (int j = 0; j < alignments_size; j++){
			score[b*alignments_size+j] = 0; 
			for (int k = 0; k < query_depth; k++) {
				int score_idx = b*alignments_size+j;
				int query_idx = b*query_depth+k;
				int key_idx = b*query_depth*alignments_size+k*alignments_size+j;
				score[score_idx] += query[query_idx] * keys[key_idx];  
			}
					
		}  	
	}
	return score;
}


template <typename Dtype>
Dtype* LuongAttentionLayer<Dtype>::luong_score(const Dtype* query,Dtype* memory) {
	cout<<"Test 5:luong_score"<<endl;
	Dtype* score = matmul(query, memory);
	softmax(score);
	return score;
}


template <typename Dtype>
Dtype* LuongAttentionLayer<Dtype>::transpose(const Dtype *memory){
	Dtype *_memory =
		(Dtype*)malloc(batch_size * query_depth * alignments_size * sizeof(Dtype));

	cout<<"test 4:transpose"<<endl;
    for(int b=0;b<batch_size;b++)
	for(int i=0;i<alignments_size;i++)
    for(int j=0;j<query_depth;j++){
		int idx = b*query_depth*alignments_size+i*query_depth+j;
		int _idx = b*query_depth*alignments_size+j*alignments_size+i;
		_memory[_idx] = memory[idx];
	}
	return _memory;
	
}

template <typename Dtype>
Dtype* LuongAttentionLayer<Dtype>::calculate_attention(const Dtype *query,const Dtype *state) {
  cout<<"Test 2:Calculate_attention"<<endl;
 
  Dtype* alignment =
		(Dtype *)malloc(batch_size * alignments_size * sizeof(Dtype));
		
		 Dtype* _memory=transpose(this->keys);
		 alignment = luong_score(query, _memory);
  return alignment;
}
 

template <typename Dtype>
void LuongAttentionLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
			
  const Dtype *query = bottom[0]->cpu_data();
  const Dtype *state = bottom[1]->cpu_data();
  const Dtype *memory = bottom[2]->cpu_data();
  this->keys = memory;
  
  Dtype* alignment = calculate_attention(query, state);
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  const int count = top[0]->count();
  caffe_copy(count, alignment, top_data);

}

INSTANTIATE_CLASS(LuongAttentionLayer);
REGISTER_LAYER_CLASS(LuongAttention);

} // namespace caffe
