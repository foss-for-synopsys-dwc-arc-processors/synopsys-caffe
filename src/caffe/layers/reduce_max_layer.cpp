#include <algorithm>
#include <vector>

#include "caffe/layers/reduce_max_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void ReduceMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const ReduceMaxParameter& reduce_max_param = this->layer_param_.reduce_max_param();
		reduce_max_keepdims_ = reduce_max_param.keepdims();
		reduce_max_axis_.clear();
		std::copy(reduce_max_param.axis().begin(),
			reduce_max_param.axis().end(),
			std::back_inserter(reduce_max_axis_));
		axis_dim_ = reduce_max_axis_.size();
		CHECK_LE(axis_dim_, bottom[0]->num_axes()) << "the dimension of axis should be less or equal than input dimension!";
		for (int i = 0; i < axis_dim_; ++i) {
			reduce_max_axis_[i] = bottom[0]->CanonicalAxisIndex(reduce_max_axis_[i]);  
		}
		std::sort(reduce_max_axis_.begin(), reduce_max_axis_.end());   
	}

	template <typename Dtype>
	void ReduceMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int num_axes = bottom[0]->num_axes();
		vector<int> top_shape = bottom[0]->shape();
		vector<int> bottom_shape = bottom[0]->shape();
		if (reduce_max_keepdims_) {
			if (axis_dim_ != 0) {
				// has keepdims and axis
				for (int i = 0; i < axis_dim_; ++i) {
					top_shape[reduce_max_axis_[i]] = 1;
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
					top_shape.erase(top_shape.begin() + reduce_max_axis_[i]);
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
	void ReduceMaxLayer<Dtype>::InReduceMax(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int b_idx, int lv_in, int t_idx) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		vector<int> shape_in(reduce_max_axis_.size(), 0);
		for (int i = 0; i < reduce_max_axis_.size(); ++i) {
			shape_in[i] = bottom[0]->shape()[reduce_max_axis_[i]];
		}
		for (int i = 0; i < shape_in[lv_in]; ++i) {
			int b_idx_add = i * bottom[0]->count(reduce_max_axis_[lv_in] + 1);
			if (lv_in == shape_in.size() - 1) {
				if (top_data[t_idx] < bottom_data[b_idx + b_idx_add]) {
					top_data[t_idx] = bottom_data[b_idx + b_idx_add];
				}
			}
			if (lv_in < shape_in.size() - 1) {
				InReduceMax(bottom, top, b_idx + b_idx_add, lv_in + 1, t_idx);
			}
		}
	};


	template <typename Dtype>
	void ReduceMaxLayer<Dtype>::OutReduceMax(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int lv_out, int b_idx, int lv_in, int t_idx) {
		// parameters: axis_dim_, shape_out, idx_out
		vector<int> shape_out = bottom[0]->shape();
		vector<int> idx_out(bottom[0]->num_axes(), 0);
		for (int i = 0; i < idx_out.size(); ++i) {
			idx_out[i] = i;
		}
		for (int i = axis_dim_ - 1; i > -1; --i) {
			shape_out.erase(shape_out.begin() + reduce_max_axis_[i]);
			idx_out.erase(idx_out.begin() + reduce_max_axis_[i]);
		}

		// main part 
		for (int i = 0; i < shape_out[lv_out]; ++i) {
			int b_idx_add = i * bottom[0]->count(idx_out[lv_out] + 1);
			int t_idx_add = i * count_shape(shape_out, lv_out + 1);
			if (lv_out == shape_out.size() - 1) {
				InReduceMax(bottom, top, b_idx + b_idx_add, lv_in, t_idx + t_idx_add);
			}
			if (lv_out < shape_out.size() - 1) {
				OutReduceMax(bottom, top, lv_out + 1, b_idx + b_idx_add, lv_in, t_idx + t_idx_add);
			}
		}
	};

	template <typename Dtype>
	void ReduceMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// get shape_in_count and shape_out_count
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int bottom_count = bottom[0]->count();
		const int top_count = top[0]->count();
		// initialize the top_data
		std::vector<Dtype> bottom_sort(bottom_count, 0);
		for (int i = 0; i < bottom_count; ++i) {
			bottom_sort[i] = bottom_data[i];
		}
		std::sort(bottom_sort.begin(), bottom_sort.end());
		const float alpha = bottom_sort[0];
		caffe_set(top_count, Dtype(alpha), top_data);

		if (axis_dim_ == 0 || axis_dim_ == bottom[0]->num_axes()) {
			// no axis, compare all elements
			for (int i = 0; i < bottom_count; ++i) {
				if (top_data[0] < bottom_data[i]) {
					top_data[0] = bottom_data[i];
				}
			}
		}
		else {
			// has axis, compare all elements in dim:reduce_max_axis_
			int lv_out = 0;
			int lv_in = 0;
			int b_idx = 0;
			int t_idx = 0;
			OutReduceMax(bottom, top, lv_out, b_idx, lv_in, t_idx);
		}
	}

	INSTANTIATE_CLASS(ReduceMaxLayer);
	REGISTER_LAYER_CLASS(ReduceMax);

}  // namespace caffe
