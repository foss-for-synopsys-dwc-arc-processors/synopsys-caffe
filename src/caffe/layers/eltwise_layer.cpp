#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      != EltwiseParameter_EltwiseOp_SUM
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Eltwise layer only takes coefficients for summation.";
  CHECK(!(this->layer_param().eltwise_param().operation() ==
	  EltwiseParameter_EltwiseOp_DIV && bottom.size() != 2)) <<
      "Eltwise layer only accepts 2 inputs for division.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  input_scale_ = vector<Dtype>(bottom.size(), 1.0);
  if (this->layer_param().eltwise_param().input_scale_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      input_scale_[i] = this->layer_param().eltwise_param().input_scale(i);
    }
  }
  input_zero_point_ = vector<int>(bottom.size(), 0);
  if (this->layer_param().eltwise_param().input_zero_point_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      input_zero_point_[i] = this->layer_param().eltwise_param().input_zero_point(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
  output_scale_ = this->layer_param_.eltwise_param().output_scale();
  output_zero_point_ = this->layer_param_.eltwise_param().output_zero_point();
  saturate_ = this->layer_param_.eltwise_param().saturate();

  //<--CUSTOMIZATION, for broadcasting
  const EltwiseParameter& param = this->layer_param_.eltwise_param();
  if(bottom.size() > 1 && bottom[0]->num_axes() > 0)
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  else axis_ = 0;

  //CUSTOMIZATION-->
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //for (int i = 1; i < bottom.size(); ++i) {
  //  CHECK(bottom[0]->shape() == bottom[i]->shape())
  //      << "bottom[0]: " << bottom[0]->shape_string()
  //      << ", bottom[" << i << "]: " << bottom[i]->shape_string();
  //}

  //<--CUSTOMIZATION, add support for broadcasting
  //const EltwiseParameter& param = this->layer_param_.eltwise_param();
  //axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  if(bottom.size() > 1){
	Blob<Dtype>* eltwise = bottom[1];
	CHECK_GE(bottom[0]->num_axes(), axis_ + eltwise->num_axes())
	<< "eltwise blob's shape extends past bottom[0]'s shape when applied "
	<< "starting with bottom[0] axis = " << axis_;
	for (int i = 0; i < eltwise->num_axes(); ++i) {
	  CHECK_EQ(bottom[0]->shape(axis_ + i), eltwise->shape(i))
					<< "dimension mismatch between bottom[0]->shape(" << axis_ + i
					<< ") and eltwise->shape(" << i << ")";
	}
	outer_dim_ = bottom[0]->count(0, axis_);
	eltwise_dim_ = eltwise->count();
	inner_dim_ = bottom[0]->count(axis_ + eltwise->num_axes());
	dim_ = eltwise_dim_ * inner_dim_;
	eltwise_multiplier_.Reshape(vector<int>(1, inner_dim_));
	if (eltwise_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)) {
	  caffe_set(inner_dim_, Dtype(1), eltwise_multiplier_.mutable_cpu_data());
	}
  }
  //CUSTOMIZATION-->

  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
	  EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
	max_idx_.Reshape(bottom[0]->shape());
  }
  //<--CUSTOMIZATION
  if (this->layer_param_.eltwise_param().operation() ==
	  EltwiseParameter_EltwiseOp_MIN && top.size() == 1) {
	min_idx_.Reshape(bottom[0]->shape());
  }
  //CUSTOMIZATION-->
}

template <typename Dtype>
int affine_and_shift(const Dtype x, const int zp_in, const Dtype mul, const int shift) {
  int r = (int) std::round((x - zp_in) * mul);
  r = r << shift;
  return r;
}

template <typename Dtype>
void tflite_add_kernel(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
  const vector<Dtype> &input_scale, const vector<int> &input_zero_point,
  const Dtype &output_scale, const int &output_zero_point) {
  /*** This kernel is not yet tested with tf-lite models
    Refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/add.cc#L172-L184
  ***/
  const int shift = 20;
  const Dtype twice_max_scale = std::max(input_scale[0], input_scale[1]) * 2.0;
  const Dtype mul_x = input_scale[0] / twice_max_scale;
  const Dtype mul_y = input_scale[1] / twice_max_scale;
  const Dtype mul_z = twice_max_scale / output_scale;
  const Dtype* x_data = bottom[0]->cpu_data();
  const Dtype* y_data = bottom[1]->cpu_data();
  Dtype* z_data = top[0]->mutable_cpu_data();

  const int N = top[0]->count();
  int affine_x, affine_y, affine_z;
  for (int i = 0; i < N; ++i) {
    affine_x = affine_and_shift(x_data[i], input_zero_point[0], mul_x, shift);
    affine_y = affine_and_shift(y_data[i], input_zero_point[1], mul_y, shift);
    affine_z = affine_x + affine_y;
    affine_z = (int) std::round(mul_z * affine_z);
    affine_z = affine_z >> shift;
    affine_z += output_zero_point;
    z_data[i] = Dtype(affine_z);
  }
}

template <typename Dtype>
void caffe2_int8add_kernel(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
  const vector<Dtype> &input_scale, const vector<int> &input_zero_point,
  const Dtype &output_scale, const int &output_zero_point) {
  // refer to https://github.com/pytorch/pytorch/pull/14089#issuecomment-439545562
  Dtype max_scale = std::max(input_scale[0], input_scale[1]) / output_scale;
  const int max_22bits = 1 << 21;
  int shift = 0;
  Dtype a_multiplier = input_scale[0] / output_scale;
  Dtype b_multiplier = input_scale[1] / output_scale;
  while (max_scale < max_22bits) {
    // the result will be 2^22 <= max_scale < 2^23, cast to integer it will occupy 22 bits
    max_scale *= 2;
    a_multiplier *= 2;
    b_multiplier *= 2;
    ++shift;
  }
  const int a_mul = (int) std::round(a_multiplier);
  const int b_mul = (int) std::round(b_multiplier);

  const Dtype* a_data = bottom[0]->cpu_data();
  const Dtype* b_data = bottom[1]->cpu_data();
  Dtype* z_data = top[0]->mutable_cpu_data();
  const int N = top[0]->count();
  long long int acc; // the accumulator
  // use output_multiplier to simulate right shift
  // expected right-shift uses rounding half to infinity
  const int out_mul = 1 << shift;
  for (int i = 0; i < N; ++i) {
    acc = 0;
    acc += ( (long long int) std::round(a_data[i]) - input_zero_point[0]) * a_mul;
    acc += ( (long long int) std::round(b_data[i]) - input_zero_point[1]) * b_mul;
    // right shift operation on a two's complement representation does not round toward zero
    // use division instead
    z_data[i] = std::round(Dtype(acc) / out_mul);
    z_data[i] += output_zero_point;
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  bool is_quant = false;
  for (int i = 0; i < bottom.size(); ++i) {
    is_quant |= (input_scale_[i] != Dtype(1.0) || input_zero_point_[i] != 0);
  }
  is_quant |= (output_scale_ != Dtype(1.0) || output_zero_point_ != 0);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* eltwise_data = NULL; //CUSTOMIZATION
  if(bottom.size() > 1)
    eltwise_data = bottom[1]->cpu_data(); //CUSTOMIZATION
  int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_DIV: //CUSTOMIZATION, assume only 2 inputs exist
	if(bottom[0]->shape() != bottom[1]->shape()){ //need broadcasting
	  for (int n = 0; n < outer_dim_; ++n) {
	    for (int d = 0; d < eltwise_dim_; ++d) {
	  	  const Dtype factor = Dtype(1)/eltwise_data[d]; //turn the division into scaling
	  	  caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
	  	  bottom_data += inner_dim_;
	  	  top_data += inner_dim_;
	  	}
	  }
	}
	else
      caffe_div(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    break;
  case EltwiseParameter_EltwiseOp_PROD:
	//<--CUSTOMIZATION
	if(bottom.size() == 1){
	  caffe_copy(count, bottom_data, top_data);
      break;
	}
	if(bottom.size() >1 && bottom[0]->shape() != bottom[1]->shape()){ //need broadcasting
	  for (int n = 0; n < outer_dim_; ++n) {
	    for (int d = 0; d < eltwise_dim_; ++d) {
	      const Dtype factor = eltwise_data[d];
	      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
	      bottom_data += inner_dim_;
	      top_data += inner_dim_;
	    }
	  }
	}
	else{
	  caffe_mul(count, bottom_data, eltwise_data, top_data);
	}
	//CUSTOMIZATION-->
	top_data = top[0]->mutable_cpu_data();
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    if (is_quant) {
      // introduce custom computation
      caffe2_int8add_kernel(bottom, top, input_scale_, input_zero_point_, output_scale_, output_zero_point_);
      break;
    }
    caffe_set(count, Dtype(0), top_data);
	//<--CUSTOMIZATION
	caffe_axpy(count, coeffs_[0],  bottom_data, top_data);
	if(bottom.size() == 1)
      break;
	if(bottom.size() >1 && bottom[0]->shape() != bottom[1]->shape()){ //need broadcasting
	  for (int n = 0; n < outer_dim_; ++n) {
	    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, eltwise_dim_,
	        inner_dim_, 1, coeffs_[1], eltwise_data,
	        eltwise_multiplier_.cpu_data(), Dtype(1), top_data);
	    top_data += dim_;
	  }
	}
	else
	  caffe_axpy(count, coeffs_[1],  eltwise_data, top_data);
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
	top_data = top[0]->mutable_cpu_data();
	//CUSTOMIZATION-->
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize
    mask = max_idx_.mutable_cpu_data();
    caffe_set(count, -1, mask);
    caffe_set(count, Dtype(-FLT_MAX), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
    //<--CUSTOMIZATION
	if(bottom.size() == 1){
	  caffe_copy(count, bottom_data, top_data);
      break;
	}
    if(bottom.size() >1 && bottom[0]->shape() != bottom[1]->shape()){ //need broadcasting
      for (int n = 0; n < outer_dim_; ++n) {
        for (int d = 0; d < eltwise_dim_; ++d) {
    	  for (int t = 0; t < inner_dim_; ++t) {
    	    int idx = n * eltwise_dim_ * inner_dim_ + d * inner_dim_ + t;
    	    if (bottom_data_a[idx] > bottom_data_b[d]) {
    	      top_data[idx] = bottom_data_a[idx];  // maxval
    	      mask[idx] = 0;  // maxid
    	    } else {
    	      top_data[idx] = bottom_data_b[d];  // maxval
    	      mask[idx] = 1;  // maxid
    	    }
    	  }
    	}
      }
    }
    else{
    //CUSTOMIZATION-->
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_a[idx] > bottom_data_b[idx]) {
          top_data[idx] = bottom_data_a[idx];  // maxval
          mask[idx] = 0;  // maxid
        } else {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = 1;  // maxid
        }
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] > top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = blob_idx;  // maxid
        }
      }
    }
    break;
  //<--CUSTOMIZATION
  case EltwiseParameter_EltwiseOp_MIN:
    // Initialize
    mask = min_idx_.mutable_cpu_data();
    caffe_set(count, -1, mask);
    caffe_set(count, Dtype(FLT_MAX), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
	if(bottom.size() == 1){
	  caffe_copy(count, bottom_data, top_data);
      break;
	}
    if(bottom.size() >1 && bottom[0]->shape() != bottom[1]->shape()){ //need broadcasting
      for (int n = 0; n < outer_dim_; ++n) {
        for (int d = 0; d < eltwise_dim_; ++d) {
    	  for (int t = 0; t < inner_dim_; ++t) {
    	    int idx = n * eltwise_dim_ * inner_dim_ + d * inner_dim_ + t;
    	    if (bottom_data_a[idx] < bottom_data_b[d]) {
    	      top_data[idx] = bottom_data_a[idx];  // minval
    	      mask[idx] = 0;  // minid
    	    } else {
    	      top_data[idx] = bottom_data_b[d];  // minval
    	      mask[idx] = 1;  // minid
    	    }
    	  }
    	}
      }
    }
    else{
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_a[idx] < bottom_data_b[idx]) {
          top_data[idx] = bottom_data_a[idx];  // minval
          mask[idx] = 0;  // minid
        } else {
          top_data[idx] = bottom_data_b[idx];  // minval
          mask[idx] = 1;  // minid
        }
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] < top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // minval
          mask[idx] = blob_idx;  // minid
        }
      }
    }
    break;
    //CUSTOMIZATION-->
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }

  if (saturate_ == EltwiseParameter_SaturateMethod_Signed)
    caffe_cpu_signed_saturate(count, top_data);
  if (saturate_ == EltwiseParameter_SaturateMethod_Unsigned)
    caffe_cpu_unsigned_saturate(count, top_data);
  if (saturate_ == EltwiseParameter_SaturateMethod_Signed_8bit)
    caffe_cpu_signed_8bit_saturate(count, top_data);
  if (saturate_ == EltwiseParameter_SaturateMethod_Unsigned_8bit)
    caffe_cpu_unsigned_8bit_saturate(count, top_data);
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_DIV:
        NOT_IMPLEMENTED;
        break;
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_mul(count, bottom[j]->cpu_data(), bottom_diff,
                        bottom_diff);
            }
          }
        } else {
          caffe_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_cpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Dtype gradient = 0;
          if (mask[index] == i) {
            gradient += top_diff[index];
          }
          bottom_diff[index] = gradient;
        }
        break;
      case EltwiseParameter_EltwiseOp_MIN:
        mask = min_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Dtype gradient = 0;
          if (mask[index] == i) {
            gradient += top_diff[index];
          }
          bottom_diff[index] = gradient;
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(EltwiseLayer);
REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe
