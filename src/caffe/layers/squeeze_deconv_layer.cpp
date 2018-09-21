/***************************** MulticoreWare_Modified - Feature: Pruning / Splicing ************************************/
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layers/squeeze_deconv_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void SqueezeDeconvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer <Dtype>::LayerSetUp(bottom, top);

  SqueezeConvolutionParameter sqconv_param = this->layer_param_.squeeze_convolution_param();

  if(this->blobs_.size()==2 && (this->bias_term_)){
    //Resized the blobs to store WeightMask,BiasMask, mu and std values
    this->blobs_.resize(5);

    // Intialize and fill the weightmask & biasmask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        sqconv_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        sqconv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());
    //Blob is set to the shape (1,1,1,2) to store mu and std values alone
    this->blobs_[4].reset(new Blob<Dtype>(1,1,1,2));
  }
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->blobs_.resize(3);
    // Intialize and fill the weightmask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        sqconv_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[1].get());
    this->blobs_[2].reset(new Blob<Dtype>(1,1,1,2));
  }

  // Intializing the tmp tensor
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  if(this->bias_term_)
    this->bias_tmp_.Reshape(this->blobs_[1]->shape());

  // Intialize the hyper-parameters
  this->std = 0;this->mu = 0;
  this->gamma = sqconv_param.gamma();
  this->power = sqconv_param.power();
  this->crate = sqconv_param.c_rate();
  this->iter_stop_ = sqconv_param.iter_stop();
  this->dynamicsplicing = sqconv_param.dynamicsplicing();
  this->splicing_rate = sqconv_param.splicing_rate();
}

template <typename Dtype>
void SqueezeDeconvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int pad_l = this->pad_l_; //CUSTOMIZATION
  const int pad_r = this->pad_r_; //CUSTOMIZATION
  const int pad_t = this->pad_t_; //CUSTOMIZATION
  const int pad_b = this->pad_b_; //CUSTOMIZATION
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    int output_dim;
    //<--CUSTOMIZATIONs
    if (pad_l!=0 || pad_r!=0 || pad_t!=0 || pad_b!=0){ //only support 2D
      if (i==0) {
        output_dim = stride_data[i] * (input_dim - 1)
            + kernel_extent - pad_t - pad_b;
      }
      if (i==1) {
        output_dim = stride_data[i] * (input_dim - 1)
            + kernel_extent - pad_l - pad_r;
      }
    }
    else{
      output_dim = stride_data[i] * (input_dim - 1)
          + kernel_extent - 2 * pad_data[i];
    }
    //CUSTOMIZATION-->
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void SqueezeDeconvolutionLayer<Dtype>::AggregateParams(const int n, const Dtype* wb, const Dtype* mask,
    unsigned int *count) {
  for (unsigned int k = 0;k < n; ++k) {
    this->mu  += fabs(mask[k] * wb[k]);
    this->std += mask[k] * wb[k] * wb[k];
    if (mask[k] * wb[k] != 0) (*count)++;
  }
}

template <typename Dtype>
void SqueezeDeconvolutionLayer<Dtype>::CalculateMask(const int n, const Dtype* wb, Dtype* mask,
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
void SqueezeDeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  vector<int> index_zero;
    // Core logic for Pruning/Splicing
  if (this->phase_ == TRAIN) {
    //To avoid corrupted mask value
    for (int l =0; l<maskcount; l++)
    {
      if (weightMask[l] !=0 && weightMask[l]!= 1)
      {
        weightMask[l] = abs(round(weightMask[l]));
      }
    }
    // Calculate the mean and standard deviation of learnable parameters
    if ((this->std == 0 && this->iter_ == 0) || this->iter_== 40 || this->iter_== 80 || this->iter_== 120 || this->iter_== 160) {
      unsigned int ncount = 0;
      AggregateParams(this->blobs_[0]->count(), weight, weightMask, &ncount);
      if (this->bias_term_) {
        AggregateParams(this->blobs_[1]->count(), bias, biasMask, &ncount);
      }
      this->mu /= ncount; this->std -= ncount *this->mu * this->mu;
      this->std /= ncount; this->std = sqrt(this->std);
      // Storing mu and std values into the blob
      prune_threshold_params[0] = this->mu;
      prune_threshold_params[1] = this->std;
    }
    // No pruning/splicing during Retraining
    // Calculate the weight mask and bias mask with probability
    Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    if (pow(1 + (this->gamma) * (this->iter_), -(this->power)) > r && (this->iter_) < (this->iter_stop_)) {
      CalculateMask(this->blobs_[0]->count(), weight, weightMask, prune_threshold_params[0], prune_threshold_params[1], this->crate);
      if (this->bias_term_) {
        CalculateMask(this->blobs_[1]->count(), bias, biasMask, prune_threshold_params[0], prune_threshold_params[1], this->crate);
      }
    }


// Dynamic Splicing
// Randomly unprune the pruned weights based on the splicing ratio
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
          start_index = zero_count - to_bespliced;  //Update start index
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
    weightTmp[k] = weight[k]*weightMask[k];
  }
  if (this->bias_term_){
    for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
      biasTmp[k] = bias[k]*biasMask[k];
    }
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_cpu_gemm(bottom_data + bottom[i]->offset(n), weightTmp,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        //const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), biasTmp);
      }
    }
  }
}

template <typename Dtype>
void SqueezeDeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weightTmp = this->weight_tmp_.cpu_data();
  const Dtype* weightMask = NULL;
  if(this->bias_term_)
    weightMask = this->blobs_[2]->cpu_data();
  else
    weightMask = this->blobs_[1]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i =0; i< top.size(); ++i)
  {
    const Dtype* top_diff = top[i]->cpu_diff();
    //const Dtype* bottom_data = bottom_data[i]->cpu_data();
    //Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
        Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
        const Dtype* biasMask = this->blobs_[3]->cpu_data();
        for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
            bias_diff[k] = bias_diff[k]*biasMask[k];
        }
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]){
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
            weight_diff[k] = weight_diff[k]*weightMask[k];
        }
        for(int n = 0; n < this->num_; ++n){
            // Gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
                this->weight_cpu_gemm(top_diff + top[i]->offset(n),
                    bottom_data + bottom[i]->offset(n), weight_diff);
            }
            // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
            // we might have just computed above.
            if (propagate_down[i]) {
                this->forward_cpu_gemm(top_diff + top[i]->offset(n), weightTmp,
                    bottom_diff + bottom[i]->offset(n),
                    this->param_propagate_down_[0]);
            }
        }
    }
  }
}
#ifdef CPU_ONLY
STUB_GPU(SqueezeDeconvolutionLayer);
#endif

INSTANTIATE_CLASS(SqueezeDeconvolutionLayer);
REGISTER_LAYER_CLASS(SqueezeDeconvolution);
}// namespace caffe
/***********************************************************************************************************************/
