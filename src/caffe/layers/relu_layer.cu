#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

#define SIGNED_SATURATE_MAX 2047
#define SIGNED_SATURATE_MIN -2048
#define UNSIGNED_SATURATE_MAX 4095
#define SIGNED_8BIT_SATURATE_MAX 127
#define SIGNED_8BIT_SATURATE_MIN -128
#define UNSIGNED_8BIT_SATURATE_MAX 255

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype relu6, Dtype maximum, Dtype minimum, Dtype saturate) { //CUSTOMIZATION
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
    if(relu6) //CUSTOMIZATON
      out[index] = out[index] > 6 ? 6: out[index]; //CUSTOMIZATON
    if(maximum > Dtype(0)) //CUSTOMIZATON
      out[index] = out[index] > maximum ? maximum: out[index]; //CUSTOMIZATON
    if(minimum != Dtype(0)) //CUSTOMIZATON
      out[index] = out[index] < minimum ? minimum: out[index]; //CUSTOMIZATON

    //<--CUSTOMIZATION
    if(saturate ==  ReLUParameter_SaturateMethod_Signed){
      if(out[index] < 0) //only need to do the round when multiplied with negative_slope
        out[index] = rint(out[index]);
      if(out[index] > SIGNED_SATURATE_MAX)
        out[index] = SIGNED_SATURATE_MAX;
      if(out[index] < SIGNED_SATURATE_MIN)
        out[index] = SIGNED_SATURATE_MIN;
    }
    if(saturate ==  ReLUParameter_SaturateMethod_Signed_8bit){
      if(out[index] < 0) //only need to do the round when multiplied with negative_slope
        out[index] = rint(out[index]);
      if(out[index] > SIGNED_8BIT_SATURATE_MAX)
        out[index] = SIGNED_8BIT_SATURATE_MAX;
      if(out[index] < SIGNED_8BIT_SATURATE_MIN)
        out[index] = SIGNED_8BIT_SATURATE_MIN;
    }
    if(saturate ==  ReLUParameter_SaturateMethod_Unsigned){
      if(out[index] > UNSIGNED_SATURATE_MAX)
        out[index] = UNSIGNED_SATURATE_MAX;
    }
    if(saturate ==  ReLUParameter_SaturateMethod_Unsigned_8bit){
      if(out[index] > UNSIGNED_8BIT_SATURATE_MAX)
        out[index] = UNSIGNED_8BIT_SATURATE_MAX;
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
  Dtype maximum = this->layer_param_.relu_param().maximum(); //CUSTOMIZATION
  Dtype minimum = this->layer_param_.relu_param().minimum(); //CUSTOMIZATION
  if (bottom.size() > 1)  //bottom[1] provides the maximum case
  	maximum = bottom[1]->gpu_data()[0];
  Dtype saturate = this->layer_param_.relu_param().saturate(); //CUSTOMIZATION
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope, relu6, maximum, minimum, saturate); //CUSTOMIZATION
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope,
    Dtype relu6, Dtype maximum, Dtype minimum) { //CUSTOMIZATON
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
    if(relu6) //CUSTOMIZATION
      out_diff[index] *= (in_data[index] < Dtype(6));
    if(maximum > Dtype(0)) //CUSTOMIZATION
      out_diff[index] *= (in_data[index] < maximum);
    if(minimum != Dtype(0)) //CUSTOMIZATION
      out_diff[index] *= (in_data[index] > minimum);
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
    Dtype maximum = this->layer_param_.relu_param().maximum(); //CUSTOMIZATION
    Dtype minimum = this->layer_param_.relu_param().minimum(); //CUSTOMIZATION
    if (bottom.size() > 1)  //bottom[1] provides the maximum case
      maximum = bottom[1]->gpu_data()[0];
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope, relu6, maximum, minimum);  //CUSTOMIZATION
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
