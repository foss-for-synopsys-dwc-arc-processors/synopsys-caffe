#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

//<--CUSTOMIZATION
template <typename Dtype>
__global__ void MinForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype minval = FLT_MAX;
    int minidx = -1;
    if (bottom_data_a[index] < bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        minval = bottom_data_a[index];
        top_data[index] = minval;
        minidx = blob_idx;
        mask[index] = minidx;
      }
    } else {
      minval = bottom_data_b[index];
      top_data[index] = minval;
      minidx = blob_idx + 1;
      mask[index] = minidx;
    }
  }
}
//CUSTOMIZATION-->

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_DIV:
	caffe_gpu_div(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
	break;
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
    case EltwiseParameter_EltwiseOp_SUM:
    //<--CUSTOMIZATION
    for (int i = 0; i < bottom.size(); ++i) {
      // input = (bottom - ZeroPoint) * scale; scale is given as coeffs_
      if (input_zero_point_[i] != 0) {
        caffe_gpu_add_scalar(count, Dtype(-input_zero_point_[i]), bottom[i]->mutable_gpu_data());
      }
    }
    //CUSTOMIZATION-->
    caffe_gpu_set(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    //<--CUSTOMIZATION
    if (output_scale_ != Dtype(1)) {
      caffe_gpu_scal(count, output_scale_, top_data);
      caffe_gpu_round(count, top_data);
    }
    // output =  top/scale + ZeroPoint
    if (output_zero_point_ != 0) {
      caffe_gpu_add_scalar(count, Dtype(output_zero_point_), top_data);
    }
    if(saturate_ ==  EltwiseParameter_SaturateMethod_Signed)
      caffe_gpu_signed_saturate(count, top_data);
    if(saturate_ ==  EltwiseParameter_SaturateMethod_Unsigned)
      caffe_gpu_unsigned_saturate(count, top_data);
    if(saturate_ ==  EltwiseParameter_SaturateMethod_Signed_8bit)
      caffe_gpu_signed_8bit_saturate(count, top_data);
    if(saturate_ ==  EltwiseParameter_SaturateMethod_Unsigned_8bit)
      caffe_gpu_unsigned_8bit_saturate(count, top_data);
    // shift the bottom blob back, in case they are input of some other residual connection
    for (int i = 0; i < bottom.size(); ++i) {
      if (input_zero_point_[i] != 0) {
        caffe_gpu_add_scalar(count, Dtype(input_zero_point_[i]), bottom[i]->mutable_gpu_data());
      }
    }
    //CUSTOMIZATION-->
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
    }
    break;
  //<--CUSTOMIZATION
  case EltwiseParameter_EltwiseOp_MIN:
    mask = min_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MinForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
    }
    break;
    //CUSTOMIZATION-->
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

//<--CUSTOMIZATION
template <typename Dtype>
__global__ void MinBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}
//CUSTOMIZATION-->

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
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
              caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                            bottom_diff);
            }
          }
        } else {
          caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.gpu_data();
        MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, i, mask, bottom_diff);
        break;
      //<--CUSTOMIZATION
      case EltwiseParameter_EltwiseOp_MIN:
        mask = min_idx_.gpu_data();
        MinBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, i, mask, bottom_diff);
        break;
        //CUSTOMIZATION-->
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
