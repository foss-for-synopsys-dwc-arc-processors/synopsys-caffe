#include <algorithm>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>


#include "caffe/layers/attention_layer.hpp"
using namespace std;
namespace caffe {    

	template <typename Dtype>
	void  AttentionLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top){

		keys = NULL;
		batch_size = this->layer_param_.attention_param().batch_size();        
		query_depth = this->layer_param_.attention_param().query_depth();
		alignments_size = this->layer_param_.attention_param().alignments_size();
		scale = this->layer_param_.attention_param().scale();
		normalize = this->layer_param_.attention_param().normalize();
		attention_option = this->layer_param_.attention_param().attention_option();
	}

	template <typename Dtype>
	void AttentionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,const vector<Blob<Dtype> *> &top) {
		vector<int> top_shape = {batch_size,alignments_size};
		top[0]->Reshape(top_shape);
	}


	template <typename Dtype>
	void AttentionLayer<Dtype>::softmax(Dtype* score){
		
		for(int b=0;b<batch_size;b++){	
			Dtype max = *std::max_element(score+b*alignments_size, score+b*alignments_size+alignments_size);
			Dtype sum = 0.0;
			

			for (int i = 0; i < alignments_size; i++) {
				score[b*alignments_size+i] = (score[b*alignments_size+i]-max);
				score[b*alignments_size+i] = (Dtype)exp(score[b*alignments_size+i]);
				sum += score[b*alignments_size+i];
			}
			

			for (int i = 0; i < alignments_size; i++){
				score[b*alignments_size+i] = (Dtype) (score[b*alignments_size+i]/(Dtype)sum);
			}	
		}
	}


	template <typename Dtype>
	Dtype* AttentionLayer<Dtype>::matmul(const Dtype* query,Dtype* keys){
		
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
	Dtype* AttentionLayer<Dtype>::luong_score(const Dtype* query,Dtype* memory) {
		Dtype* score = matmul(query, memory);
		softmax(score);
		return score;
	}


	template <typename Dtype>
	Dtype* AttentionLayer<Dtype>::bahdanau_score(const Dtype* query,Dtype* memory, Dtype* attention_v, Dtype attention_g, Dtype* attention_b) {

		Dtype* normed_v =
		(Dtype*)malloc(query_depth* sizeof(Dtype));
		
		Dtype* score =
		(Dtype*)malloc(batch_size * alignments_size * sizeof(Dtype));
		
		
		Dtype* result =
		(Dtype*)malloc(batch_size * query_depth * alignments_size * sizeof(Dtype));
		
		Dtype sum =0.0;
		if(attention_g && attention_b){
			for(int i=0;i<query_depth;i++){
				Dtype x  = sqrt(attention_v[i] <=0 ? 0 : attention_v[i]);
				sum+=x;
			}

			for(int i=0;i<query_depth;i++){
				normed_v[i]=attention_g*attention_v[i]/sqrt(sum);
			}
			
			for(int b =0; b < batch_size ; b++)
			for(int i =0; i < alignments_size ; i++)
			for(int j =0; j < query_depth ; j++){
				int idx = b*alignments_size*query_depth+i*query_depth+j;
				result[idx] = keys[idx]+query[b*query_depth+j]+attention_b[j];
				result[idx] = normed_v[j]*tanh(result[idx]);
			}	
		}else{
			
			for(int b =0; b < batch_size ; b++)
			for(int i =0; i < alignments_size ; i++)
			for(int j =0; j < query_depth ; j++){
				
				int idx = b*alignments_size*query_depth+i*query_depth+j;
				result[idx] = keys[idx]+query[b*query_depth+j];
				result[idx] = attention_v[j]*tanh(result[idx]);
			}		
		}
		
		for(int b =0; b < batch_size ; b++)
		for(int i =0; i < alignments_size ; i++){
			Dtype score_ =0;
			for(int j =0; j < query_depth ; j++){
				int idx = b*alignments_size*query_depth+i*query_depth+j;
				score_+= result[idx];
			}
			score[b*alignments_size+i] = score_;
		}
		
		softmax(score);
		return score;
	}

	template <typename Dtype>
	Dtype* AttentionLayer<Dtype>::transpose(const Dtype *memory){
		Dtype *_memory =
		(Dtype*)malloc(batch_size * query_depth * alignments_size * sizeof(Dtype));

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
	Dtype* AttentionLayer<Dtype>::calculate_attention(const Dtype *query,const Dtype *state) {
		
		Dtype* alignment = (Dtype *)malloc(batch_size * alignments_size * sizeof(Dtype));
		
		Dtype* _memory=transpose(this->keys);



		/*luong_score*/		
		// alignment = luong_score(query, _memory);
		
		/*bahdanau_score*/		 
		/*	 
		attention_v=torch.Tensor(1,self.query_depth)
		attention_v=nn.init.xavier_uniform_(attention_v)
		*/

		Dtype* attention_v = (Dtype*) malloc(query_depth * sizeof(Dtype));
		for(int i=0;i<query_depth;i++)
		attention_v[i] = this->attention_v[i];


		Dtype attention_g  = sqrt(1.0/query_depth);
		Dtype* attention_b = new Dtype[query_depth];
        for(int i=0;i<query_depth;i++)
			attention_b[i]=0.0;
		
	
		
		//alignment = bahdanau_score(query, _memory,attention_v, NULL, NULL);

		alignment = bahdanau_score(query, _memory,attention_v, attention_g, attention_b);
		
		return alignment;
	}


	template <typename Dtype>
	void AttentionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
		
		const Dtype *query = bottom[0]->cpu_data();
		const Dtype *state = bottom[1]->cpu_data();
		const Dtype *memory = bottom[2]->cpu_data();
		const Dtype *attention_v = bottom[3]->cpu_data();

		this->keys = memory;
		this->attention_v = attention_v;

		Dtype* alignments = calculate_attention(query, state);
		
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_set(top[0]->count(), Dtype(0.), top_data);
		const int count = top[0]->count();
		caffe_copy(count, alignments, top_data);

	}

	INSTANTIATE_CLASS(AttentionLayer);
	REGISTER_LAYER_CLASS(Attention);
} // namespace caffe
