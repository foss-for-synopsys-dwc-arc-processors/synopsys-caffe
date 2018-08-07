/***************************** MulticoreWare_Modified - Feature: Pruning / Splicing ************************************/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/squeeze_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>

namespace caffe {

// The constant NUM_THREADS should be equal to the value in SqueezeCMomentCalc
template <typename Dtype>
__global__ void SqueezeCMomentCollect(const int n, const Dtype* wb, const Dtype* mask,
    Dtype* mu, Dtype* std, unsigned int* count ) {
  const int NUM_THREADS = 512;
  __shared__ Dtype param [4*NUM_THREADS];
  __shared__ unsigned int tcount [2*NUM_THREADS];
  unsigned int t = threadIdx.x;
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  if (s+t < n){
    param[t] = fabs(mask[s + t] * wb[s + t]);
    param[t + 2 * NUM_THREADS] = mask[s + t] * wb[s + t] * wb[s + t];
    if(mask[s + t] * wb[s + t]!= 0) tcount[t] = 1;
    else tcount[t] = 0;
  }
  else{
    param[t] = 0;param[t +2 * NUM_THREADS] = 0;tcount[t] = 0;
  }
  if (s + t + NUM_THREADS < n){
    param[t + NUM_THREADS] = fabs(mask[s + t + NUM_THREADS] * wb[s + t + NUM_THREADS]);
    param[t + 3 * NUM_THREADS] = mask[s + t + NUM_THREADS] * wb[ s + t + NUM_THREADS] *wb[s + t + NUM_THREADS];
    if(mask[s + t + NUM_THREADS] * wb[s + t + NUM_THREADS] != 0) tcount[t + NUM_THREADS] = 1;
    else tcount[t + NUM_THREADS] = 0;
  }
  else{
    param[t + NUM_THREADS] = 0;param[t + 3 * NUM_THREADS] = 0;tcount[t + NUM_THREADS] = 0;
  }
  __syncthreads();
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      param[t] += param[t + stride];
      param[t + 2 * NUM_THREADS] += param[t + 2 * NUM_THREADS + stride];
      tcount[t] += tcount[t+stride];
    }
    __syncthreads();
  }
  if (t == 0){
    mu   [blockIdx.x] = param[0];
    std  [blockIdx.x] = param[2*NUM_THREADS];
    count[blockIdx.x] = tcount[0];
  }
}

// The constant NUM_THREADS should be equal to the value in SqueezeCMomentCalc
template <typename Dtype>
__global__ void SqueezeCNzeroCollect(const int n, const Dtype* mask, unsigned int* count ) {
  const int NUM_THREADS = 512;
  __shared__ unsigned int tcount [2 * NUM_THREADS];
  unsigned int t = threadIdx.x;
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  tcount[t] = 0;
  if (s + t < n && mask[s+t]!=0){
    tcount[t] = 1;
  }
  tcount[t + NUM_THREADS] = 0;
  if (s + t + NUM_THREADS < n && mask[s + t + NUM_THREADS] != 0){
    tcount[t + NUM_THREADS] = 1;
  }
  __syncthreads();
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      tcount[t] += tcount[t+stride];
    }
    __syncthreads();
  }
  if (t == 0){
    count[blockIdx.x] = tcount[0];
  }
}
//Condition for pruning and splicing
template <typename Dtype>
__global__ void SqueezeCMaskCalc(const int n, const Dtype* wb,
    Dtype* mask, Dtype mu, Dtype std, Dtype r) {
  CUDA_KERNEL_LOOP(index, n) {
    // The constants 0.9 and 1.1 is to set margin that witholds few parameters undergoing pruning / splicing
    if (mask[index] > 0 && fabs(wb[index]) <= 0.9 * r * max(mu + std, Dtype(0)))
      mask[index] = 0;
    else if (mask[index] == 0 && fabs(wb[index])> 1.1 * r * max(mu + std, Dtype(0)) && r!=0)
      mask[index] = 1;
  }
}

template <typename Dtype>
__global__ void SqueezeCMaskApply(const int n, const Dtype* wb,
    const Dtype* mask, Dtype* wb_t) {
  CUDA_KERNEL_LOOP(index, n) {
    wb_t[index] = wb[index] * mask[index];
  }
}

template <typename Dtype>
__global__ void ValidateMask(const int n,  Dtype* wb) {
  CUDA_KERNEL_LOOP(index, n) {
  if (wb[index] !=0 && wb[index]!= 1)
    wb[index] = fabs(rintf(wb[index]));
  }
}

template <typename Dtype>
void SqueezeCMomentCalc(const int n, const Dtype* wb, const Dtype* mask, Dtype* mu, Dtype* std, unsigned int* ncount){
  const unsigned int NUM_THREADS = 512;
  Dtype* pmu_g; Dtype* pstd_g; unsigned int* pncount_g;
  Dtype* pmu_c; Dtype* pstd_c; unsigned int* pncount_c;
  int num_p = (n+(NUM_THREADS<<1)-1)/(NUM_THREADS<<1);
  cudaMalloc(&pmu_g, sizeof(Dtype)  * num_p);
  cudaMalloc(&pstd_g, sizeof(Dtype) * num_p);
  cudaMalloc(&pncount_g, sizeof(unsigned int) * num_p);
  pmu_c = (Dtype*) malloc(num_p * sizeof(Dtype));
  pstd_c = (Dtype*) malloc(num_p * sizeof(Dtype));
  pncount_c = (unsigned int*) malloc(num_p * sizeof(unsigned int));
  SqueezeCMomentCollect<Dtype><<<num_p,NUM_THREADS>>>(n, wb, mask, pmu_g, pstd_g, pncount_g);
  CUDA_POST_KERNEL_CHECK;
  cudaMemcpy(pmu_c, pmu_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pstd_c, pstd_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pncount_c, pncount_g, sizeof(unsigned int) * num_p, cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_p; i++) {
    *mu += pmu_c[i];*std += pstd_c[i];*ncount += pncount_c[i];
  }
  cudaFree(pmu_g);cudaFree(pstd_g);cudaFree(pncount_g);
  free(pmu_c);free(pstd_c);free(pncount_c);
}

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = NULL;
  Dtype* weightMask = NULL;
  Dtype* weightTmp = NULL;
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL;
  Dtype* prune_threshold_params_gpu = NULL; // To store mu and std values
  Dtype* prune_threshold_params_cpu = NULL;
  prune_threshold_params_cpu = (Dtype*)malloc(sizeof(Dtype) * 2);
  int maskcount = 0;
  if (this->bias_term_) {
    weight = this->blobs_[0]->mutable_gpu_data();
    weightMask = this->blobs_[2]->mutable_gpu_data();
    weightTmp = this->weight_tmp_.mutable_gpu_data();
    bias = this->blobs_[1]->mutable_gpu_data();
    biasMask = this->blobs_[3]->mutable_gpu_data();
    prune_threshold_params_gpu = this->blobs_[4]->mutable_gpu_data();
    biasTmp = this->bias_tmp_.mutable_gpu_data();
    maskcount = this->blobs_[2]->count();
  }
  else {
    weight = this->blobs_[0]->mutable_gpu_data();
    weightMask = this->blobs_[1]->mutable_gpu_data();
    prune_threshold_params_gpu = this->blobs_[2]->mutable_gpu_data();
    weightTmp = this->weight_tmp_.mutable_gpu_data();
    maskcount = this->blobs_[1]->count();
  }

  if (this->phase_ == TRAIN){
    // Validate mask value to avoid corrupted mask value
    ValidateMask<Dtype><<<CAFFE_GET_BLOCKS(maskcount),
    CAFFE_CUDA_NUM_THREADS>>>( maskcount, weightMask);
    CUDA_POST_KERNEL_CHECK;

    // Calculate the mean and standard deviation of learnable parameters
    if (this->std == 0 && this->iter_ == 0){
      unsigned int ncount = 0;
      SqueezeCMomentCalc(this->blobs_[0]->count(), weight, weightMask, &this->mu, &this->std, &ncount);
      if (this->bias_term_) {
        SqueezeCMomentCalc(this->blobs_[1]->count(), bias, biasMask, &this->mu, &this->std, &ncount);
      }
      this->mu /= ncount; this->std -= ncount * this->mu * this->mu;
      this->std /= ncount; this->std = sqrt(this->std);
      prune_threshold_params_cpu[0] = this->mu;
      prune_threshold_params_cpu[1] = this->std;
      LOG(INFO)<<mu<<"  "<<std<<"  "<<ncount<<"\n";
      // Copy mu and std value from host to device
      cudaMemcpy(prune_threshold_params_gpu, prune_threshold_params_cpu, sizeof(Dtype)*2, cudaMemcpyHostToDevice);
    }
    // Copy mu and std value from Device to host
    cudaMemcpy(prune_threshold_params_cpu, prune_threshold_params_gpu, sizeof(Dtype)*2, cudaMemcpyDeviceToHost);
    // No pruning done during Retraining
    // Calculate the weight mask and bias mask with probability
    Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    if (pow(1+(this->gamma)*(this->iter_),-(this->power))>r && (this->iter_)<(this->iter_stop_)) {
      SqueezeCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, prune_threshold_params_cpu[0], prune_threshold_params_cpu[1], this->crate);
      CUDA_POST_KERNEL_CHECK;
      if (this->bias_term_) {
        SqueezeCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
          CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, prune_threshold_params_cpu[0], prune_threshold_params_cpu[1], this->crate);
        CUDA_POST_KERNEL_CHECK;
      }
    }
    // Dynamic Splicing
    // Unprune the pruned weights based on the splicing ratio
    if(this->dynamicsplicing)
    {
      if (this->iter_ == 0) {
        Dtype* weight_cpu = (Dtype *)malloc(this->blobs_[0]->count() *(sizeof(Dtype)));
        Dtype* weightMask_cpu = (Dtype *)malloc(this->blobs_[0]->count() *(sizeof(Dtype)));
        // Initially copy weight, weightMask to weight_cpu, weightMask_cpu and do Dynamic Splicing
        cudaMemcpy(weight_cpu, weight, this->blobs_[0]->count() *(sizeof(Dtype)), cudaMemcpyDeviceToHost);
        cudaMemcpy(weightMask_cpu, weightMask, this->blobs_[0]->count() *(sizeof(Dtype)), cudaMemcpyDeviceToHost);
        // Vector Pair holds weights and corresponding index for pruned nodes
        std::vector<std::pair<float, int> > prune_node;
        for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
          if(weightMask_cpu[k] == 0) {
            prune_node.push_back(make_pair(fabs(weight_cpu[k]), k));
          }
        }
        // Sort the weights and unprune the nodes
        std::sort(prune_node.begin(), prune_node.end());
        int zero_count = prune_node.size();
        int to_bespliced = zero_count * this->splicing_rate;
        int start_index = 0;
        int end_index = 0;
        for (unsigned int k = 0; k < zero_count; ++k) {
          if (prune_node[k].first > (0.25 * (prune_threshold_params_cpu[0] + prune_threshold_params_cpu[1]))) {
            start_index = k;
            break;
          }
        }
        if(start_index == 0)
          start_index = zero_count - to_bespliced; //Update start index
        end_index = start_index + to_bespliced;
        if (end_index > zero_count) {
          start_index = start_index - (end_index - zero_count);
          end_index = start_index + to_bespliced;
        }
        for (unsigned int k = start_index; k < end_index; ++k) {
          weightMask_cpu[prune_node[k].second] = 1;
        }
        cudaMemcpy(weightMask, weightMask_cpu, this->blobs_[0]->count() *(sizeof(Dtype)), cudaMemcpyHostToDevice);
        free(weightMask_cpu);
        free(weight_cpu);
        this->dynamicsplicing = false;
      }
    }
    free(prune_threshold_params_cpu);
  }

  // Calculate the current (masked) weight and bias
  SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
    CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
  CUDA_POST_KERNEL_CHECK;
  if (this->bias_term_) {
    SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, biasTmp);
    CUDA_POST_KERNEL_CHECK;
  }

  // Forward calculation with (masked) weight and bias
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                        weightTmp, bottom_data, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            biasTmp, top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weightTmp, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            biasTmp, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype* weightMask = NULL;
    if(this->bias_term_)
      weightMask = this->blobs_[2]->gpu_data();
    else
      weightMask = this->blobs_[1]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight_diff, weightMask, weight_diff);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* biasMask = this->blobs_[3]->gpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    // Gradient with respect to bias
    SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[3]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[3]->count(), bias_diff, biasMask, bias_diff);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,bias_diff);
  }
  if (propagate_down[0]) {
    const Dtype* weightTmp = this->weight_tmp_.gpu_data();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weightTmp, (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SqueezeInnerProductLayer);

}  // namespace caffe
/***********************************************************************************************************************/
