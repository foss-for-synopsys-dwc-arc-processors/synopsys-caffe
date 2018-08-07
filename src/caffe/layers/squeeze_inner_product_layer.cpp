/***************************** MulticoreWare_Modified - Feature: Pruning / Splicing ************************************/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/squeeze_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.squeeze_inner_product_param().num_output();
  bias_term_ = this->layer_param_.squeeze_inner_product_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.squeeze_inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.squeeze_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.squeeze_inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  SqueezeInnerProductParameter squeeze_inner_param = this->layer_param_.squeeze_inner_product_param();

  if(this->blobs_.size() == 2 && (this->bias_term_)){
    //Resized the blobs to store WeightMask,BiasMask, mu and std values
    this->blobs_.resize(5);
    // Intialize and fill the weightmask & biasmask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        squeeze_inner_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        squeeze_inner_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());
    //Blob is set to the shape (1,1,1,2) to store mu and std values alone
    this->blobs_[4].reset(new Blob<Dtype>(1,1,1,2));
  }
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->blobs_.resize(3);
    // Intialize and fill the weightmask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        squeeze_inner_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[1].get());
    this->blobs_[2].reset(new Blob<Dtype>(1,1,1,2));
  }
  // Intialize the tmp tensor 
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  if(this->bias_term_)
    this->bias_tmp_.Reshape(this->blobs_[1]->shape());

  // Intialize the hyper-parameters
  this->std = 0;this->mu = 0;  
  this->gamma = squeeze_inner_param.gamma();
  this->power = squeeze_inner_param.power();
  this->crate = squeeze_inner_param.c_rate();
  this->iter_stop_ = squeeze_inner_param.iter_stop();
  this->dynamicsplicing = squeeze_inner_param.dynamicsplicing();
  this->splicing_rate = squeeze_inner_param.splicing_rate();
}

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::AggregateParams(const int n, const Dtype* wb, const Dtype* mask,
    unsigned int *count) {

  for (unsigned int k = 0;k < n; ++k) {
    this->mu  += fabs(mask[k] * wb[k]);
    this->std += mask[k] * wb[k] * wb[k];
    if (mask[k] * wb[k] != 0) (*count)++;
  }
}

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::CalculateMask(const int n, const Dtype* wb, Dtype* mask,
    Dtype mu, Dtype std, Dtype r) {

  for (unsigned int k = 0;k < n;++k) {
        // The constants 0.9 and 1.1 is to set margin that witholds few parameters undergoing pruning / splicing
        if (mask[k] > 0 && fabs(wb[k]) <= 0.9 * r *  std::max(mu + std, Dtype(0)))
          mask[k] = 0; //Pruning
        else if (mask[k] == 0 && fabs(wb[k]) > 1.1 * r * std::max(mu + std, Dtype(0)) && r != 0)
          mask[k] = 1; //Splicing
      }
}

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* weight = NULL;
  Dtype* weightMask = NULL;
  Dtype* weightTmp = NULL;
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL;
  Dtype* prune_threshold_params = NULL;
  int maskcount = 0;
  if (this->bias_term_) {
    weight = this->blobs_[0]->mutable_cpu_data();
    weightMask = this->blobs_[2]->mutable_cpu_data();
    weightTmp = this->weight_tmp_.mutable_cpu_data();
    bias = this->blobs_[1]->mutable_cpu_data();
    biasMask = this->blobs_[3]->mutable_cpu_data();
    prune_threshold_params = this->blobs_[4]->mutable_cpu_data(); // To store mu and std values
    biasTmp = this->bias_tmp_.mutable_cpu_data();
    maskcount = this->blobs_[2]->count();
  }
  else {
    weight = this->blobs_[0]->mutable_cpu_data();
    weightMask = this->blobs_[1]->mutable_cpu_data();
    prune_threshold_params = this->blobs_[2]->mutable_cpu_data();
    weightTmp = this->weight_tmp_.mutable_cpu_data();
    maskcount = this->blobs_[1]->count();
  }

  if (this->phase_ == TRAIN){
    // To avoid corrupted mask value
    for (int l =0; l<maskcount; l++)
    {
      if (weightMask[l] !=0 && weightMask[l]!= 1)
      {
        weightMask[l] = abs(round(weightMask[l]));
      }
    }
    // Calculate the mean and standard deviation of learnable parameters 
    if (this->std==0 && this->iter_==0) {
      unsigned int ncount = 0;
      AggregateParams(this->blobs_[0]->count(), weight, weightMask, &ncount);
      if (this->bias_term_) {
        AggregateParams(this->blobs_[1]->count(), bias, biasMask, &ncount);
      }
      // Storing mu and std values into the blob
      prune_threshold_params[0] = this->mu;
      prune_threshold_params[1] = this->std;
    }
    // No pruning done during Retraining
    // Perform pruning or Splicing
    Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    if (pow( 1 + (this->gamma) * (this->iter_), -(this->power)) > r && (this->iter_) < (this->iter_stop_)) {
      CalculateMask(this->blobs_[0]->count(), weight, weightMask, prune_threshold_params[0],prune_threshold_params[1], this->crate);
      if (this->bias_term_) {
        CalculateMask(this->blobs_[1]->count(), bias, biasMask, prune_threshold_params[0], prune_threshold_params[1], this->crate);
      }
    }
    // Dynamic Splicing
    // Unprune the pruned weights based on the splicing ratio
    if(this->dynamicsplicing)
    {
      if (this->iter_ == 0) {
        // Vector Pair holds weights and corresponding index for pruned nodes
        std::vector<std::pair<float, int> > prune_node;
        for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
          if(weightMask[k] == 0) {
            prune_node.push_back(make_pair(fabs(weight[k]), k));
          }
        }
        // Sort the weights and unprune the nodes
        std::sort(prune_node.begin(), prune_node.end());
        int zero_count = prune_node.size();
        int to_bespliced = zero_count * this->splicing_rate;
        int start_index = 0;
        int end_index = 0;
        for (unsigned int k = 0; k < zero_count; ++k) {
          if (prune_node[k].first > (0.25 * (prune_threshold_params[0] + prune_threshold_params[1]))) {
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
          weightMask[prune_node[k].second] = 1;
        }
        this->dynamicsplicing = false;
      }
    }
  }

  // Calculate the current (masked) weight and bias
  for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
    weightTmp[k] = weight[k] * weightMask[k];
  }
  if (this->bias_term_){
    for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
      biasTmp[k] = bias[k] * biasMask[k];
    }
  }

  // Forward calculation with (masked) weight and bias 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weightTmp, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(), biasTmp, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void SqueezeInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Use the masked weight to propagate back
  const Dtype* top_diff = top[0]->cpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype* weightMask = NULL;
    if(this->bias_term_)
      weightMask = this->blobs_[2]->cpu_data();
    else
      weightMask = this->blobs_[1]->cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
      weight_diff[k] = weight_diff[k]*weightMask[k];
    }
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., weight_diff);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* biasMask = this->blobs_[3]->cpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    // Gradient with respect to bias
    for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
      bias_diff[k] = bias_diff[k]*biasMask[k];
    }
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1., bias_diff);
  }
  if (propagate_down[0]) {
    const	Dtype* weightTmp = this->weight_tmp_.cpu_data();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weightTmp, (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(SqueezeInnerProductLayer);
#endif

INSTANTIATE_CLASS(SqueezeInnerProductLayer);
REGISTER_LAYER_CLASS(SqueezeInnerProduct);

}  // namespace caffe
/***********************************************************************************************************************/
