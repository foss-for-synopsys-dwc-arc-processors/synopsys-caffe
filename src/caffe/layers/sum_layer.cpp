#include <algorithm>
#include <vector>

#include "caffe/layers/sum_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const SumParameter& sum_param = this->layer_param_.sum_param();

		sum_keepdims_ = sum_param.keepdims();

		sum_axis_.clear();
		std::copy(sum_param.axis().begin(),
			sum_param.axis().end(),
			std::back_inserter(sum_axis_));
		axis_dim_ = sum_axis_.size();
		for (int i = 0; i < axis_dim_; ++i) {
			sum_axis_[i] = bottom[0]->CanonicalAxisIndex(sum_axis_[i]);   
		}
		std::sort(sum_axis_.begin(), sum_axis_.end());
	}

	template <typename Dtype>
	void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int num_axes = bottom[0]->num_axes();

		vector<int> top_shape = bottom[0]->shape();
		vector<int> bottom_shape = bottom[0]->shape();

		if (sum_keepdims_) {
			if (axis_dim_ != 0) {
				// has keepdims and axis
				for (int i = 0; i < axis_dim_; ++i) {
					top_shape[sum_axis_[i]] = 1;
				}
			}
			else {
				// has keepdims but no axis
				for (int i = 0; i < num_axes; ++i) {
					top_shape[i] = 1;
				}
			}
		}
		else {
			if (axis_dim_ != 0) {
				// no keepdims but has axis
				for (int i = axis_dim_ - 1; i > -1; --i) {
					top_shape.erase(top_shape.begin() + sum_axis_[i]);
				}
			}
			else {
				// no axis and no keepdims
				top_shape.resize(0);
			}
		}
		top[0]->Reshape(top_shape);
	}


	template <typename Dtype>
	void SumLayer<Dtype>::InSum(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int b_idx, int lv_in, int t_idx, vector<int> idx_in) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		vector<int> shape_in(idx_in.size(), 0);
		for (int i = 0; i < idx_in.size(); ++i) {
			shape_in[i] = bottom[0]->shape()[idx_in[i]];
		}

		for (int i = 0; i < shape_in[lv_in]; ++i) {
			int b_idx_add = i * bottom[0]->count(idx_in[lv_in] + 1);
			if (lv_in == shape_in.size() - 1) {
				//const int sum_size = bottom[0]->count(sum_axis_[axis_dim_ - 1] + 1);
				//caffe_axpy(Dtype(1), Dtype(1), bottom_data + b_idx + b_idx_add, top_data + t_idx);
				top_data[t_idx] += bottom_data[b_idx + b_idx_add];
			}
			if (lv_in < shape_in.size() - 1) {
				InSum(bottom, top, b_idx + b_idx_add, lv_in + 1, t_idx, idx_in);
			}
		}
	};


	template <typename Dtype>
	void SumLayer<Dtype>::OutSum(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int lv_out, int b_idx, int lv_in, int t_idx, vector<int> sum_axis_) {

		// parameters: shape_out, idx_out
		const int axis_dim_ = sum_axis_.size();
		vector<int> shape_out = bottom[0]->shape();
		vector<int> idx_out(bottom[0]->num_axes(), 0);
		for (int i = 0; i < idx_out.size(); ++i) {
			idx_out[i] = i;
		}
		//shape_out.resize(sum_axis_[axis_dim_ - 1] + 1);
		for (int i = axis_dim_ - 1; i > -1; --i) {
			shape_out.erase(shape_out.begin() + sum_axis_[i]);
			idx_out.erase(idx_out.begin() + sum_axis_[i]);
		}

		// main part 

		for (int i = 0; i < shape_out[lv_out]; ++i) {
			int b_idx_add = i * bottom[0]->count(idx_out[lv_out] + 1);
			int t_idx_add = i * count_shape(shape_out, lv_out + 1);
			if (lv_out == shape_out.size() - 1) {
				
				InSum(bottom, top, b_idx + b_idx_add, lv_in, t_idx + t_idx_add, sum_axis_);

			}
			if (lv_out < shape_out.size() - 1) {
				OutSum(bottom, top, lv_out + 1, b_idx + b_idx_add, lv_in, t_idx + t_idx_add, sum_axis_);
			}
		}
	};

	template <typename Dtype>
	void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		const int count = top[0]->count();
		const int bottom_count = bottom[0]->count();


		if (axis_dim_ == 0) {
			// no axis, add all elements
			for (int i = 0; i < bottom_count; ++i) {
				top_data[0] += bottom_data[i];
			}
		}
		else {
			// has axis, add all elements in dim:sum_axis_
			int lv_out = 0;
			int lv_in = 0;
			int b_idx = 0;
			int t_idx = 0;

			OutSum(bottom, top, lv_out, b_idx, lv_in, t_idx, sum_axis_);
		}
	}

	INSTANTIATE_CLASS(SumLayer);
	REGISTER_LAYER_CLASS(Sum);

}  // namespace caffe
