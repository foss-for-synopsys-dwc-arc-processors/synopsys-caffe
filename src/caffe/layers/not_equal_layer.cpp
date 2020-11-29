#include <math.h>
#include <vector>

#include "caffe/layers/not_equal_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void NotEqualLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {
		if ((bottom.size() == 1) && (this->blobs_.size() == 0)) {
			const NotEqualParameter &not_equal_param = this->layer_param_.not_equal_param();
			comparand_ = not_equal_param.comparand();
			const_flag_ = 1;
		}
		else {
			const_flag_ = 0;
			Blob<Dtype>* comparand = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
			// case 1: bottom[0] and bottom[1] are tensor and have the same dimension
			if (comparand->num_axes() == bottom[0]->num_axes()) {
				for (int i = 0; i < bottom[0]->num_axes(); ++i) {
					CHECK_EQ(bottom[0]->shape(i), comparand->shape(i)) << "Broadcasting is not supported now!!! Please confirm that 2 inputs have the same shape!!";
				}
			}
			// case 2: bottom[0] and bottom[1] are tensor and have different dimension/shape
			else {
				CHECK_EQ(bottom[0]->num_axes(), comparand->num_axes()) << "Broadcasting is not supported now!!! Please confirm that 2 inputs have the same shape!!";
			}
		}
	}

	template <typename Dtype>
	void NotEqualLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> bottom_shape = bottom[0]->shape();
		top[0]->Reshape(bottom_shape);
	}

	template <typename Dtype>
	void NotEqualLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		if (const_flag_ == 1) {
			for (int i = 0; i < top[0]->count(); ++i) {
				top_data[i] = bool(bottom_data[i] != comparand_);
			}
		}
		else {
			Blob<Dtype>* comparand = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
			const Dtype* comparand_data = comparand->cpu_data();
			for (int i = 0; i < top[0]->count(); ++i) {
				top_data[i] = bool(bottom_data[i] != comparand_data[i]);
			}
		}
	}

INSTANTIATE_CLASS(NotEqualLayer);
REGISTER_LAYER_CLASS(NotEqual);

} // namespace caffe

