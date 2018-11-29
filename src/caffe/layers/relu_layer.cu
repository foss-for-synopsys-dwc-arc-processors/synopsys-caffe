#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

#define SIGNED_SATURATE_MAX 2047
#define SIGNED_SATURATE_MIN -2048
#define UNSIGNED_SATURATE_MAX 4095

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype relu6, Dtype saturate) { //CUSTOMIZATION
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
    if(relu6) //CUSTOMIZATON
      out[index] = out[index] > 6 ? 6: out[index]; //CUSTOMIZATON
    //<--CUSTOMIZATION
    if(saturate ==  ReLUParameter_SaturateMethod_Signed){
      if(out[index] < 0) //only need to do the round when multiplied with negative_slope
        out[index] = rint(out[index]);
      if(out[index] > SIGNED_SATURATE_MAX)
        out[index] = SIGNED_SATURATE_MAX;
      if(out[index] < SIGNED_SATURATE_MIN)
        out[index] = SIGNED_SATURATE_MIN;
    }
    if(saturate ==  ReLUParameter_SaturateMethod_Unsigned){
      if(out[index] > UNSIGNED_SATURATE_MAX)
        out[index] = UNSIGNED_SATURATE_MAX;
    }
    //CUSTOMIZATION-->
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  Dtype relu6 = this->layer_param_.relu_param().relu6(); //CUSTOMIZATION
  Dtype saturate = this->layer_param_.relu_param().saturate(); //CUSTOMIZATION
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope, relu6, saturate); //CUSTOMIZATION
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope, Dtype relu6) { //CUSTOMIZATON
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
    if(relu6) //CUSTOMIZATION
      out_diff[index] *= (in_data[index] < Dtype(6));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    Dtype relu6 = this->layer_param_.relu_param().relu6(); //CUSTOMIZATION
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope, relu6);  //CUSTOMIZATION
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
