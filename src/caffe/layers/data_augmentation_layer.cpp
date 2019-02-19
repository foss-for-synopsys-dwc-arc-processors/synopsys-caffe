// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/data_augmentation_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>

using std::max;
using std::min;

namespace caffe {

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}  

inline float clamp(float f, float a, float b) {
    return fmaxf(a, fminf(f, b));
}


template <typename Dtype>
void DataAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  // TODO This won't work when applying a net to images of size different from what the net was trained on
  aug_ = this->layer_param_.augmentation_param();
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "DataAugmentationLayer only runs Reshape on setup";
  if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping data mean blob initialization";
  else {
    if (aug_.recompute_mean()) {
      LOG(INFO) << "Recompute mean";
      this->blobs_.resize(3);
      this->blobs_[1].reset(new Blob<Dtype>());
      this->layer_param_.add_param();
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_lr_mult(0.);
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_decay_mult(0.);
      this->blobs_[2].reset(new Blob<Dtype>());
      this->layer_param_.add_param();
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_lr_mult(0.);
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_decay_mult(0.);      
    } 
    else {  
      LOG(INFO) << "Do not recompute mean";
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));      
    // Never backpropagate
    this->param_propagate_down_.resize(this->blobs_.size(), false);
    this->layer_param_.add_param();
    this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_lr_mult(0.);
    this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_decay_mult(0.); 
//     LOG(INFO) << "DEBUG: this->layer_param_.param_size()=" << this->layer_param_.param_size();
//     LOG(INFO) << "DEBUG: Writing layer_param";
#if !defined(_MSC_VER)
    WriteProtoToTextFile(this->layer_param_, "/misc/lmbraid17/sceneflownet/dosovits/matlab/test/message.prototxt");
#endif
//     LOG(INFO) << "DEBUG: Finished writing layer_param";
  } 
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    LOG(WARNING) << "Reshape of Augmentation layer should only be called once? Check this";
    CHECK_GE(bottom.size(), 1) << "Data augmentation layer takes one or two input blobs.";
    CHECK_LE(bottom.size(), 2) << "Data augmentation layer takes one or two input blobs.";
    CHECK_GE(top.size(), 1) << "Data augmentation layer outputs one or two output blobs.";
    CHECK_LE(top.size(), 2) << "Data augmentation layer outputs one or two output blobs.";

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

    output_params_=(top.size()>1);
    input_params_=(bottom.size()>1);
    aug_ = this->layer_param_.augmentation_param();
    discount_coeff_schedule_ = this->layer_param_.coeff_schedule_param();

    // Dimensions
    do_cropping_ = (aug_.has_crop_width() && aug_.has_crop_height());
    if (!do_cropping_)
    {
        cropped_width_ = width;
        cropped_height_ = height;
        LOG(WARNING) << "Please enter crop size if you want to perform augmentation";
    }
    else
    {
        cropped_width_ = aug_.crop_width();    CHECK_GE(width, cropped_width_)   << "crop width greater than original";
        cropped_height_ = aug_.crop_height();  CHECK_GE(height, cropped_height_) << "crop height greater than original";
    }

    // Allocate output
    top[0]->Reshape(num, channels, cropped_height_, cropped_width_);

    // Coeff stuff
    AugmentationCoeff coeff;
    num_params_ = coeff.GetDescriptor()->field_count();

    // If this layer is given coefficients from another augmentation layer, take this blob (same ref)
    if (input_params_) {
        LOG(INFO) << "Receiving " << num_params_ << " augmentation params";
        all_coeffs_.ReshapeLike(*bottom[1]);
    } else
        all_coeffs_.Reshape(num, num_params_, 1, 1); //create

    if (output_params_) {
        top[1]->ReshapeLike(all_coeffs_);
        LOG(INFO) << "Emitting " << num_params_ << " augmentation params";
    }

    // Coeff transformation matrix cache for one batch
    coeff_matrices_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tTransMat)));
    
    coeff_chromatic_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tChromaticCoeffs)));
    coeff_chromatic_eigen_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tChromaticEigenCoeffs)));
    coeff_effect_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tEffectCoeffs)));

    chromatic_eigenspace_.reset(new SyncedMemory(sizeof(typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace)));

    // Data mean computation
    if (aug_.recompute_mean()) {
      ones_.Reshape(1, 1, cropped_height_, cropped_width_);
      caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());
      this->blobs_[1]->Reshape(1, channels, cropped_height_, cropped_width_);
      this->blobs_[2]->Reshape(1, channels, 1, 1);
    }
    else if(aug_.mean().size()==3 && !aug_.mean_per_pixel())
    {
      ones_.Reshape(1, 1, cropped_height_, cropped_width_);
      caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());

      LOG(INFO) << "Using predefined per-pixel mean from proto";
      pixel_rgb_mean_from_proto_.Reshape(1,3,1,1);
      for(int i=0; i<3; i++)
          pixel_rgb_mean_from_proto_.mutable_cpu_data()[i]=aug_.mean().Get(i);
    }
    
    noise_.reset(new SyncedMemory(top[0]->count() / top[0]->num() * sizeof(Dtype)));

    *(this->blobs_[0]->mutable_cpu_data()) = 0;
    
//     LOG(INFO) << "DEBUG: Reshape done";
}


template <typename Dtype>
void DataAugmentationLayer<Dtype>::adjust_blobs(vector<Blob<Dtype>* > blobs)
{
    if (aug_.recompute_mean() > 0 && blobs.size() >= 2) {
      LOG(INFO) << "Data augmentation layer: adjusting mean blobs";
      CHECK_EQ(this->blobs_[1]->channels(), blobs[1]->shape(1));
      bool same_size = (this->blobs_[1]->width() == blobs[1]->shape(3)) &&
                       (this->blobs_[1]->height() == blobs[1]->shape(2));
      int channels = this->blobs_[1]->channels();
      int area = this->blobs_[1]->height() * this->blobs_[1]->width();
      int source_area = blobs[1]->shape(2) * blobs[1]->shape(3);
      this->blobs_[0]->CopyFrom(*blobs[0],false,true);
      LOG(INFO) << "recovered iteration count " << this->blobs_[0]->cpu_data()[0];
      if (this->layer_param().augmentation_param().mean_per_pixel()==false)
      {
             // If using RGB mean, copy only RGB values (blob 2)
             Blob<Dtype> tmp; tmp.CopyFrom(*blobs[2], false,true);
             caffe_copy(this->blobs_[2]->count(), tmp.mutable_cpu_data(), this->blobs_[2]->mutable_cpu_data());
             Dtype* data_mean_per_channel_cpu = this->blobs_[2]->mutable_cpu_data();
             for(int i=0; i<this->blobs_[2]->count(); i++)
                 LOG(INFO) << "recovered mean value " << data_mean_per_channel_cpu[i];
      }
      else
      {
          if (same_size) {
            this->blobs_[1]->CopyFrom(*blobs[1],false,true);
            caffe_cpu_gemv(CblasNoTrans, channels, area, Dtype(1)/Dtype(area),
                           this->blobs_[1]->cpu_data(), ones_.cpu_data(), Dtype(0), this->blobs_[2]->mutable_cpu_data());
          } else {
            Blob<Dtype> tmp_mean;
            Blob<Dtype> tmp_ones;
            tmp_mean.CopyFrom(*blobs[1],false,true);
            tmp_ones.Reshape(1, 1, tmp_mean.height(), tmp_mean.width());
            caffe_set(tmp_ones.count(), Dtype(1), tmp_ones.mutable_cpu_data());
            caffe_cpu_gemv(CblasNoTrans, channels, source_area, Dtype(1)/Dtype(source_area),
                           tmp_mean.cpu_data(), tmp_ones.cpu_data(), Dtype(0), this->blobs_[2]->mutable_cpu_data());
            caffe_cpu_gemm(CblasNoTrans, CblasTrans, channels, area, 1,
                           Dtype(1), this->blobs_[2]->mutable_cpu_data(), this->ones_.cpu_data(), Dtype(0), this->blobs_[1]->mutable_cpu_data());
          }
      }
    }
    if (blobs.size() < 2)
      LOG(INFO) << "Data augmentation layer: no blobs to copy";
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::ComputeChromaticEigenspace_cpu(const int num,
                                            const int channels, const int height, const int width,
                                            const Dtype* data,
                                            typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace* chromatic_eigenspace)
{
	   for (int n = 0; n < num; ++n) {
	     for (int y = 0; y < height; ++y) {
		   for (int x = 0; x < width; ++x) {

	        //int x  = index % width;
	        //int y  = (index / width) % height;
	        //int n  = (index / width / height);

	        Dtype rgb [3];
	        for (int c=0; c<channels; c++)
	          rgb[c] = data[((n*channels + c)*width + x)*height + y];

	        float mean_rgb [3]={0,0,0};
	        float max_abs_eig[3]={0,0,0};
	        float max_rgb[3]={0,0,0};
	        float min_rgb[3]={FLT_MAX,FLT_MAX,FLT_MAX};

	        Dtype eig [3];
	        for (int c=0; c<channels; c++)
	        {
	            eig[c] = chromatic_eigenspace->eigvec[3*c] * rgb[0] + chromatic_eigenspace->eigvec[3*c+1] * rgb[1] + chromatic_eigenspace->eigvec[3*c+2] * rgb[2];

	            if (fabs(eig[c]) > max_abs_eig[c])
	                max_abs_eig[c] = fabs(eig[c]);
	            if (rgb[c] > max_rgb[c])
	                max_rgb[c] = rgb[c];
	            if (rgb[c] < min_rgb[c])
	                min_rgb[c] = rgb[c];
	            mean_rgb[c] = mean_rgb[c] + rgb[c]/width/height;
	        }

	        for (int c=0; c<channels; c++) chromatic_eigenspace->mean_rgb[c] += mean_rgb[c];
	        for (int c=0; c<channels; c++) chromatic_eigenspace->max_abs_eig[c] = std::max(chromatic_eigenspace->max_abs_eig[c], max_abs_eig[c]);
	        for (int c=0; c<channels; c++) chromatic_eigenspace->max_rgb[c] = std::max(chromatic_eigenspace->max_rgb[c], max_rgb[c]);
	        for (int c=0; c<channels; c++) chromatic_eigenspace->min_rgb[c] = std::min(chromatic_eigenspace->min_rgb[c], min_rgb[c]);

   	    }
   	  }
     }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::SpatialAugmentation_cpu(const int num, const int channels, const int height, const int width,
                            const Dtype* src_data, const int src_count, const int dest_height, const int dest_width, Dtype* dest_data,
                            const typename AugmentationLayerBase<Dtype>::tTransMat *transMats)
{
	for (int cn = 0; cn < num*channels; ++cn) {
	  int n = cn / channels;
	  for (int y = 0; y < dest_height; ++y) {
		for (int x = 0; x < dest_width; ++x) {

        //int x  = index % dest_width; //w-pos
        //int y  = (index / dest_width) % dest_height; //h-pos
        //int cn = index / dest_width / dest_height; // channel*num
        //int n = cn / channels; //num

//         int c = cn % channels; // channel

        /* === Warping:
        //transMat:
        //   / 0 2 4 \
        //   \ 1 3 5 /
		*/

        float xpos = x * transMats[n].t0 + y * transMats[n].t2 + transMats[n].t4;
        float ypos = x * transMats[n].t1 + y * transMats[n].t3 + transMats[n].t5;

        xpos = clamp(xpos, 0.0f, (float)(width)-1.05f);  //Ensure that floor(xpos)+1 is still valid
        ypos = clamp(ypos, 0.0f, (float)(height)-1.05f);

        // Get interpolated sample
        //float sample = tex2DLayered(texRef, xpos, ypos, cn);
        float tlx = floor(xpos);
        float tly = floor(ypos);
        int srcIdxOff = width*(height*cn + tly) + tlx;

        float sampleTL = src_data[srcIdxOff];
        float sampleTR = src_data[min(srcIdxOff+1,src_count)];
        float sampleBL = src_data[min(srcIdxOff+width,src_count)];
        float sampleBR = src_data[min(srcIdxOff+1+width,src_count)];

        float xdist = xpos - tlx;
        float ydist = ypos - tly;

        float sample = (1-xdist)*(1-ydist)*sampleTL
                + (  xdist)*(  ydist)*sampleBR
                + (1-xdist)*(  ydist)*sampleBL
                + (  xdist)*(1-ydist)*sampleTR;

        // write sample to destination
        int index = (cn*dest_height+ y)*dest_width + x;
        dest_data[index] = sample;
    }
   }
  }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::ChromaticEigenAugmentation_cpu(const int num, const int channels, const int height, const int width,
                            Dtype* src_data, Dtype* dest_data,
                            const typename AugmentationLayerBase<Dtype>::tChromaticEigenCoeffs* chromatic,
                            typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace* eigen,
                            const float max_multiplier)
{
	   for (int n = 0; n < num; ++n) {
	     for (int y = 0; y < height; ++y) {
		   for (int x = 0; x < width; ++x) {

        //int x  = index % width;
        //int y  = (index / width) % height;
        //int n  = (index / width / height);

        Dtype s, s1, l=0, l1=0;

        // subtracting the mean
        Dtype rgb [3];
        for (int c=0; c<channels; c++)
            rgb[c] = src_data[((n*channels + c)*width + x)*height + y] - eigen->mean_rgb[c];

        // doing the nomean stuff
        Dtype eig [3];
        for (int c=0; c<channels; c++) {
            eig[c] = eigen->eigvec[3*c] * rgb[0] + eigen->eigvec[3*c+1] * rgb[1] + eigen->eigvec[3*c+2] * rgb[2];
            if ( eigen->max_abs_eig[c] > 1e-2f ) {
                eig[c] = eig[c] / eigen->max_abs_eig[c];
                if (c==0) {
                    eig[c] = copysign(pow(abs(eig[c]),(Dtype)chromatic[n].pow_nomean0),eig[c]);
                    eig[c] = eig[c] + chromatic[n].add_nomean0;
                    eig[c] = eig[c] * chromatic[n].mult_nomean0;
                }
                else if (c==1) {
                    eig[c] = copysign(pow(abs(eig[c]),(Dtype)chromatic[n].pow_nomean1),eig[c]);
                    eig[c] = eig[c] + chromatic[n].add_nomean1;
                    eig[c] = eig[c] * chromatic[n].mult_nomean1;
                } else if (c==2) {
                    eig[c] = copysign(pow(abs(eig[c]),(Dtype)chromatic[n].pow_nomean2),eig[c]);
                    eig[c] = eig[c] + chromatic[n].add_nomean2;
                    eig[c] = eig[c] * chromatic[n].mult_nomean2;
                }
            }
        }

        // re-adding the mean
        for (int c=0; c<channels; c++)
            eig[c] = eig[c] + eigen->mean_eig[c];

        // doing the withmean stuff
        if ( eigen->max_abs_eig[0] > 1e-2f) {
                eig[0] = copysign(pow(abs(eig[0]),(Dtype)chromatic[n].pow_withmean0),eig[0]);
                eig[0] = eig[0] + chromatic[n].add_withmean0;
                eig[0] = eig[0] * chromatic[n].mult_withmean0;
        }
        s = sqrt(eig[1]*eig[1] + eig[2]*eig[2]);
        s1 = s;
        if (s > 1e-2f) {
                s1 = pow(s1, (Dtype)chromatic[n].pow_withmean1);
                s1 = std::max(s1 + chromatic[n].add_withmean1, Dtype(0));
                s1 = s1 * chromatic[n].mult_withmean1;
        }
        if(chromatic[n].col_angle!=0)
        {
            Dtype temp1, temp2;
            temp1 =  cos(chromatic[n].col_angle) * eig[1] - sin(chromatic[n].col_angle) * eig[2];
            temp2 =  sin(chromatic[n].col_angle) * eig[1] + cos(chromatic[n].col_angle) * eig[2];
            eig[1] = temp1;
            eig[2] = temp2;
        }
        for (int c=0; c<channels; c++) {
            if ( eigen->max_abs_eig[c] > 1e-2f )
                eig[c] = eig[c] * eigen->max_abs_eig[c];
        }
        if (eigen->max_l > 1e-2f) {
            l1 = sqrt(eig[0]*eig[0] + eig[1]*eig[1] + eig[2]*eig[2]);
            l1 = l1 / eigen->max_l;
        }
        if (s > 1e-2f) {
            eig[1] = eig[1] / s * s1;
            eig[2] = eig[2] / s * s1;
        }
        if (eigen->max_l > 1e-2f) {
            l = sqrt(eig[0]*eig[0] + eig[1]*eig[1] + eig[2]*eig[2]);
            l1 = pow(l1, Dtype(chromatic[n].lmult_pow));
            l1 = std::max(l1 + chromatic[n].lmult_add, Dtype(0));
            l1 = l1 * chromatic[n].lmult_mult;
            l1 = l1 * eigen->max_l;
            if (l > 1e-2f)
                for (int c=0; c<channels; c++) {
                    eig[c] = eig[c] / l * l1;
                    if (eig[c] > eigen->max_abs_eig[c])
                        eig[c] = eigen->max_abs_eig[c];
                }
        }
        for (int c=0; c<channels; c++) {
            rgb[c] = eigen->eigvec[c] * eig[0] + eigen->eigvec[3+c] * eig[1] + eigen->eigvec[6+c] * eig[2];
            rgb[c] = std::min(rgb[c],Dtype(max_multiplier));
            rgb[c] = std::max(rgb[c],Dtype(0));
            dest_data[((n*channels + c)*width + x)*height + y] = rgb[c];
        }
      }
    }
  }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::ColorContrastAugmentation_cpu(const int num, const int channels, const int height, const int width,
                            Dtype* src_data, Dtype* dest_data,
                            const typename AugmentationLayerBase<Dtype>::tChromaticCoeffs* chromatic,
                            const float max_multiplier)
{
	   for (int n = 0; n < num; ++n) {
	     for (int y = 0; y < height; ++y) {
		   for (int x = 0; x < width; ++x) {

        //float x = (float)(index % width); //w-pos
        //float y = (float)((index / width) % height); //h-pos
        //int n = (index / width / height); // num

        int data_index[3];
        float rgb[3];
        float mean_in = 0;
        float mean_out = 0;

        // do color change
        for (int c=0;c<3;++c) {
          data_index[c] = width*(height*(3*n + c) + y) + x;
          rgb[c] = src_data[data_index[c]];
          mean_in += rgb[c];
          rgb[c] *= chromatic[n].color[c];
          mean_out += rgb[c];
        }

        float brightness_coeff = mean_in / (mean_out + 0.01f);

        for (int c=0;c<3;++c) {
          //compensate brightness
          rgb[c] = clamp(rgb[c] * brightness_coeff, 0.f, 1.f);

          // do gamma change
          rgb[c] = pow(rgb[c],chromatic[n].gamma);

          // do brightness change
          rgb[c] = rgb[c] + chromatic[n].brightness;

          // do contrast change
          rgb[c] = 0.5f + (rgb[c]-0.5f)*chromatic[n].contrast;

          // write sample to destination
          dest_data[data_index[c]] = clamp(rgb[c], 0.f, max_multiplier);
        }
    }
   }
  }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::ApplyEffects_cpu(const int num,
                              const int count, const int channels, const int height, const int width, Dtype* data,
                              const typename AugmentationLayerBase<Dtype>::tEffectCoeffs *effects,
                              const float max_multiplier)
{
	   for (int n = 0; n < num; ++n) {
		 for (int c = 0; c < channels; ++c) {
	       for (int y = 0; y < height; ++y) {
		     for (int x = 0; x < width; ++x) {

        //int x  = index % width; //w-pos
        //int y  = (index / width) % height; //h-pos
        //int cn = index / width / height; // channel*num
        //int n = cn / channels; //num
        //int c  = cn % channels; // channel
		int index = ((n*channels + c)*height+ y)*width + x;
        float sample=data[index];

        if((x-width/2)*effects[n].shadow_nx+(y-height/2)*effects[n].shadow_ny-effects[n].shadow_distance>0)
        {
            sample-=effects[n].shadow_strength;
        }

        data[index] = clamp(sample, 0.f, max_multiplier);
    }
   }
   }
  }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    Dtype* top_data = top[0]->mutable_cpu_data(); // dest
    int topwidth = top[0]->width();
    int topheight = top[0]->height();
    int topchannels = top[0]->channels();
    int topcount = top[0]->count();

    const Dtype* bottom_data = bottom[0]->cpu_data(); // source
    int bottomchannels = bottom[0]->channels();
    int bottomwidth = bottom[0]->width();
    int bottomheight = bottom[0]->height();
    int bottomcount = bottom[0]->count();

    int num = bottom[0]->num();
    CHECK_EQ(bottom[0]->num(), top[0]->num());

    // Debug: check for NaNs and lare values:
    const Dtype* bottom_cpu_data = bottom[0]->cpu_data();
    for(int i=0; i<bottomcount; i++) {
        if (isnan(bottom_cpu_data[i]))
            LOG(WARNING) << "bottom_data[" << i << "]=NaN";
//         if (std::fabs(bottom_cpu_data[i])>1e3)
//             LOG(WARNING) << "bottom_data[" << i << "]=" << bottom_cpu_data[i];
    }

    // Data sharing
    if (input_params_)   all_coeffs_.ShareData(*bottom[1]); //reuse
    if (output_params_)  top[1]->ShareData(all_coeffs_);

    // From bottom to top
    Dtype& num_iter = *(this->blobs_[0]->mutable_cpu_data());
    num_iter = ((int)num_iter+1);
    //     LOG(INFO) << "  augmentation: iteration " << num_iter;

    std::string write_augmented;
    if (aug_.has_write_augmented()) write_augmented = aug_.write_augmented();
    else                            write_augmented = std::string("");

    //bool augment_during_test = aug_.augment_during_test();
    bool train_phase = (this->phase_ == TRAIN);

    AugmentationParameter aug = aug_;

    if (do_cropping_) { // Only augment when cropping

        Dtype* my_params = all_coeffs_.mutable_cpu_data();
        int num_params = num_params_;
        Dtype discount_coeff = discount_coeff_schedule_.initial_coeff() +
                ( discount_coeff_schedule_.final_coeff() - discount_coeff_schedule_.initial_coeff()) *
                (Dtype(2) / (Dtype(1) + exp((Dtype)-1.0986 * num_iter / discount_coeff_schedule_.half_life())) - Dtype(1));
        //   LOG(INFO) << "num_iter=" << num_iter << ", discount_coeff=" << discount_coeff;

        if(!input_params_) {
            // If we don't have input parameters we need to generate some
            bool gen_spatial_transform         = false;
            bool gen_chromatic_transform       = false;
            bool gen_chromatic_eigen_transform = false;
            bool gen_effect_transform          = false;
            if(train_phase || aug.augment_during_test()) {
                if(aug_.has_mirror() || aug_.has_rotate() || aug_.has_zoom() || aug_.has_translate() || aug_.has_squeeze() ||         aug_.has_translate_x() || aug_.has_translate_y())
                    gen_spatial_transform   = true;
                if(aug_.has_brightness() || aug_.has_gamma() || aug_.has_contrast() || aug_.has_color())
                    gen_chromatic_transform = true;
                if(aug_.has_fog_size() || aug_.has_fog_amount() || aug_.has_motion_blur_angle() || aug_.has_motion_blur_size() || aug_.has_shadow_angle()
                        || aug_.has_shadow_distance() || aug_.has_shadow_strength() || aug_.has_noise() )
                    gen_effect_transform = true;
                if(aug_.has_lmult_pow() || aug_.has_lmult_mult() || aug_.has_lmult_add() || aug_.has_sat_pow() || aug_.has_sat_mult() || aug_.has_sat_add()
                        || aug_.has_col_pow() || aug_.has_col_mult() || aug_.has_col_add() || aug_.has_ladd_pow() || aug_.has_ladd_mult() || aug_.has_ladd_add() || aug_.has_col_rotate() )
                    gen_chromatic_eigen_transform = true;
            }

            // Preparing the coeffs:
            for (int item_id = 0; item_id < num; ++item_id)
            {
                AugmentationCoeff coeff;
                AugmentationLayerBase<Dtype>::clear_all_coeffs(coeff);

                // Sample the parameters of the transformations
                if (gen_spatial_transform)
                    AugmentationLayerBase<Dtype>::generate_valid_spatial_coeffs(aug, coeff, discount_coeff, bottomwidth, bottomheight, cropped_width_, cropped_height_, 50);

                if(gen_chromatic_transform)
                    AugmentationLayerBase<Dtype>::generate_chromatic_coeffs(aug, coeff, discount_coeff);

                if(gen_chromatic_eigen_transform)
                    AugmentationLayerBase<Dtype>::generate_chromatic_eigen_coeffs(aug, coeff, discount_coeff);

                if(gen_effect_transform)
                    AugmentationLayerBase<Dtype>::generate_effect_coeffs(aug, coeff, discount_coeff);

                if (write_augmented.size())
                {
                    if (gen_spatial_transform)
                        LOG(INFO) << "Augmenting " << item_id
                                  << ", mirror: "  << coeff.mirror()  << ", angle: " << coeff.angle()
                                  << ", zoom_x: "  << coeff.zoom_x()  << ", zoom_y: "  << coeff.zoom_y()
                                  << ", dx: "      << coeff.dx()      << ", dy: " << coeff.dy();
                    else
                        LOG(INFO) << "Not augmenting " << item_id << " spatially";

                    if (gen_chromatic_transform)
                        LOG(INFO) << "Augmenting " << item_id
                                  << ", gamma: " << coeff.gamma()
                                  << ", brightness: " << coeff.brightness()
                                  << ", contrast: " << coeff.contrast()
                                  << ", color1: " << coeff.color1()
                                  << ", color2: " << coeff.color2()
                                  << ", color3: " << coeff.color3();
                    else
                        LOG(INFO) << "Not augmenting " << item_id << " chromatically";

                    if (gen_effect_transform)
                        LOG(INFO) << "Augmenting " << item_id
                                  << ", noise: " << coeff.noise();
                    else
                        LOG(INFO) << "Not augmenting " << item_id << " with effects";
                }

                AugmentationLayerBase<Dtype>::coeff_to_array(coeff, my_params + item_id * num_params); // add new coeffs to blob
            }
        }

        // The Real work:
        typename AugmentationLayerBase<Dtype>::tTransMat *matrices = (typename AugmentationLayerBase<Dtype>::tTransMat *)(coeff_matrices_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tChromaticCoeffs *chromatics = (typename AugmentationLayerBase<Dtype>::tChromaticCoeffs*)(coeff_chromatic_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tChromaticEigenCoeffs *chromatics_eigen = (typename AugmentationLayerBase<Dtype>::tChromaticEigenCoeffs*)(coeff_chromatic_eigen_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tEffectCoeffs *effects = (typename AugmentationLayerBase<Dtype>::tEffectCoeffs*)(coeff_effect_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace *chromatic_eigen_space = (typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace*)(chromatic_eigenspace_->mutable_cpu_data());

        bool has_effect_augmentation=false;
        bool has_chromatic_augmentation=false;
        bool has_chromatic_eigen_augmentation=false;
        for (int item_id = 0; item_id < num; ++item_id)
        {
            AugmentationCoeff coeff;

            // Load the previously generated coeffs (either they are from another layer or generated above)
            AugmentationLayerBase<Dtype>::array_to_coeff(my_params + item_id * num_params, coeff);
            AugmentationLayerBase<Dtype>::clear_defaults(coeff);

            matrices[item_id].toIdentity();
            matrices[item_id].fromCoeff(&coeff,cropped_width_,cropped_height_,bottomwidth,bottomheight);

            chromatics[item_id].fromCoeff(&coeff);
            if(chromatics[item_id].needsComputation())
                has_chromatic_augmentation=true;

            chromatics_eigen[item_id].fromCoeff(&coeff);
            if(chromatics_eigen[item_id].needsComputation())
                has_chromatic_eigen_augmentation=true;

            effects[item_id].fromCoeff(&coeff);
            if(effects[item_id].needsComputation())
                has_effect_augmentation=true;

            //LOG(INFO) << "matrix " << item_id << ": " << matrices[item_id].t0 << ", " << matrices[item_id].t1 << ", " << matrices[item_id].t2 << ", " << matrices[item_id].t3 << ", " << matrices[item_id].t4 << ", " << matrices[item_id].t5;
            //LOG(INFO) << "cw/2 , ch/2: " << .5 * static_cast<float>(cropped_width_) << ", " << .5 * static_cast<float>(cropped_height_);
        }

//        LOG(INFO) << "has_effect_augmentation=" << has_effect_augmentation;
//        LOG(INFO) << "has_chromatic_augmentation=" << has_chromatic_augmentation;
//        LOG(INFO) << "has_chromatic_eigen_augmentation=" << has_chromatic_eigen_augmentation;

        if(has_chromatic_eigen_augmentation)
        {
            CHECK_EQ(bottomchannels, 3) << "Chromatic-Eigen augmentations only work with 3-channel input";
            memset(chromatic_eigen_space,0,sizeof(typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace));

            if(this->layer_param_.augmentation_param().chromatic_eigvec().size()!=9)
                LOG(ERROR) << "You need to specify chromatic eigenvectors for Chromatic-Eigen augementation";

            for(int i=0; i<9; i++)
                chromatic_eigen_space->eigvec[i]=this->layer_param_.augmentation_param().chromatic_eigvec().Get(i);

            for(int c=0; c<bottomchannels; c++)
                chromatic_eigen_space->min_rgb[c]=FLT_MAX;
        }

        typename AugmentationLayerBase<Dtype>::tTransMat *cpu_matrices = (typename AugmentationLayerBase<Dtype>::tTransMat *)(coeff_matrices_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tChromaticCoeffs *cpu_chromatics = (typename AugmentationLayerBase<Dtype>::tChromaticCoeffs*)(coeff_chromatic_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tChromaticEigenCoeffs *cpu_chromatics_eigen = (typename AugmentationLayerBase<Dtype>::tChromaticEigenCoeffs*)(coeff_chromatic_eigen_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tEffectCoeffs *cpu_effects = (typename AugmentationLayerBase<Dtype>::tEffectCoeffs*)(coeff_effect_->mutable_cpu_data());
        typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace *cpu_chromatic_eigen_space = (typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace*)(chromatic_eigenspace_->mutable_cpu_data());

        if(has_chromatic_eigen_augmentation)
        {
            CHECK_EQ(bottomchannels, 3) << "Chromatic-Eigen augmentations only work with 3-channel input";
            ComputeChromaticEigenspace_cpu(
                  num,
                  bottomchannels, bottomheight, bottomwidth,
                  bottom_data, cpu_chromatic_eigen_space);

            chromatic_eigen_space = (typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace*)(chromatic_eigenspace_->mutable_cpu_data());

            for (int c=0; c<bottomchannels; c++)
                chromatic_eigen_space->mean_rgb[c] = chromatic_eigen_space->mean_rgb[c] / num;

            for (int c=0; c<bottomchannels; c++) {
                chromatic_eigen_space->mean_eig[c] = chromatic_eigen_space->eigvec[3*c] * chromatic_eigen_space->mean_rgb[0] +
                                                     chromatic_eigen_space->eigvec[3*c+1] * chromatic_eigen_space->mean_rgb[1] +
                                                     chromatic_eigen_space->eigvec[3*c+2] * chromatic_eigen_space->mean_rgb[2];
                if (chromatic_eigen_space->max_abs_eig[c] > 1e-2 )
                    chromatic_eigen_space->mean_eig[c] = chromatic_eigen_space->mean_eig[c] / chromatic_eigen_space->max_abs_eig[c];
            }

            chromatic_eigen_space->max_l = sqrt(
                                chromatic_eigen_space->max_abs_eig[0]*chromatic_eigen_space->max_abs_eig[0] +
                                chromatic_eigen_space->max_abs_eig[1]*chromatic_eigen_space->max_abs_eig[1] +
                                chromatic_eigen_space->max_abs_eig[2]*chromatic_eigen_space->max_abs_eig[2] );


//            LOG(INFO) << "new mean_eig: " << chromatic_eigen_space->mean_eig[0] << " " << chromatic_eigen_space->mean_eig[1] << " " << chromatic_eigen_space->mean_eig[2];
//            LOG(INFO) << "new mean_rgb: " << chromatic_eigen_space->mean_rgb[0] << " " << chromatic_eigen_space->mean_rgb[1] << " " << chromatic_eigen_space->mean_rgb[2];
//            LOG(INFO) << "new max_abs_eig: " << chromatic_eigen_space->max_abs_eig[0] << " " << chromatic_eigen_space->max_abs_eig[1] << " " << chromatic_eigen_space->max_abs_eig[2];
//            LOG(INFO) << "new max_rgb: " << chromatic_eigen_space->max_rgb[0] << " " << chromatic_eigen_space->max_rgb[1] << " " << chromatic_eigen_space->max_rgb[2];
//            LOG(INFO) << "new min_rgb: " << chromatic_eigen_space->min_rgb[0] << " " << chromatic_eigen_space->min_rgb[1] << " " << chromatic_eigen_space->min_rgb[2];

//            for(int i=0; i<9; i++)
//                LOG(INFO) << "new eigvec: " <<  chromatic_eigen_space->eigvec[i];

            cpu_chromatic_eigen_space = (typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace*)(chromatic_eigenspace_->mutable_cpu_data());
        }

        SpatialAugmentation_cpu(
              num,
              bottomchannels, bottomheight, bottomwidth, bottom_data, bottomcount,
              topheight, topwidth, top_data, cpu_matrices);

        if (has_chromatic_eigen_augmentation)
        {
            CHECK_EQ(bottomchannels, 3) << "Chromatic-Eigen augmentations only work with 3-channel input";
            ChromaticEigenAugmentation_cpu(
               num,
               topchannels, topheight, topwidth, top_data, top_data, cpu_chromatics_eigen, cpu_chromatic_eigen_space, aug_.max_multiplier());
        }

        if (has_chromatic_augmentation) {
            CHECK_EQ(bottomchannels, 3) << "Chromatic augmentations only work with 3-channel input";
            ColorContrastAugmentation_cpu(
               num,
               topchannels, topheight, topwidth, top_data, top_data, cpu_chromatics, aug_.max_multiplier());
        }

        if (has_effect_augmentation)         {
            CHECK_EQ(bottomchannels, 3) << "Effect augmentations only work with 3-channel input";
            ApplyEffects_cpu(
                  num,
                  topcount, bottomchannels, topheight, topwidth, top_data,
                  cpu_effects, aug_.max_multiplier()
            );

            int count = cropped_width_*cropped_height_*bottomchannels;
            Dtype* noise_data = (Dtype*)(noise_->mutable_cpu_data());
            for (int item_id = 0; item_id < num; ++item_id) {
              if(effects[item_id].noise > 0) {
                caffe_rng_gaussian(count, Dtype(0), Dtype(effects[item_id].noise), noise_data);
                caffe_axpy(count, Dtype(1), noise_data, top_data + item_id * count);
              }
            }
        }
    } else {
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }

    // Mean subtraction stuff
    if(aug.recompute_mean() > 0 ) {
        Dtype* data_mean_cpu = this->blobs_[1]->mutable_cpu_data();
        Dtype* data_mean_per_channel_cpu = this->blobs_[2]->mutable_cpu_data();
        const Dtype* data_ones_cpu = ones_.cpu_data();
        int count = cropped_width_*cropped_height_*bottomchannels;
        int area = cropped_width_*cropped_height_;
        // Compute the mean if have not reached the max number of iterations yet
        if (num_iter <= aug.recompute_mean()) {
            CHECK_EQ(this->blobs_[1]->count(), count);
            caffe_scal(count, Dtype(num_iter-1), data_mean_cpu);
            for (int n = 0; n < num; ++n) {
                caffe_axpy(count, Dtype(1)/Dtype(num), top_data + n*count, data_mean_cpu);
            }
            caffe_scal(count, Dtype(1.) / Dtype(num_iter), data_mean_cpu);
            caffe_cpu_gemv(CblasNoTrans, bottomchannels, area, Dtype(1)/Dtype(area), data_mean_cpu, data_ones_cpu, Dtype(0), data_mean_per_channel_cpu);
        }
        // Subtract the mean from the images
        if (aug.mean_per_pixel()) { // separate mean for each pixel
            for (int n = 0; n < num; ++n) {
                caffe_axpy(count, Dtype(-1), data_mean_cpu, top_data + n*count);
            }
        } else { // only one mean for each channel
            for (int n = 0; n < num; ++n) {
//                Dtype* data_mean_per_channel_cpu = this->blobs_[2]->mutable_cpu_data();
//                for(int i=0; i<3; i++)
//                    LOG(INFO) << "subtracting RGB value " << data_mean_per_channel_cpu[i];
                caffe_cpu_gemm(CblasNoTrans, CblasTrans, bottomchannels, area, 1,
                               Dtype(-1), data_mean_per_channel_cpu, data_ones_cpu, Dtype(1), top_data + n*count);
            }
        }
    }
    else if(aug.mean().size()==3 && !aug.mean_per_pixel()) // Subtract predefined pixelwise mean
    {
        const Dtype* data_ones_cpu = ones_.cpu_data();
        Dtype* data_mean_per_channel_cpu = pixel_rgb_mean_from_proto_.mutable_cpu_data();
        int count = cropped_width_*cropped_height_*bottomchannels;
        int area = cropped_width_*cropped_height_;

        for (int n = 0; n < num; ++n) {
            caffe_cpu_gemm(CblasNoTrans, CblasTrans, bottomchannels, area, 1,
                           Dtype(-1), data_mean_per_channel_cpu, data_ones_cpu, Dtype(1), top_data + n*count);
        }
    }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	LOG(FATAL) << "DataAugmentationLayer cannot do backward.";
	return;
}

#ifdef CPU_ONLY
STUB_GPU(DataAugmentationLayer);
#endif

INSTANTIATE_CLASS(DataAugmentationLayer);
REGISTER_LAYER_CLASS(DataAugmentation);

}  // namespace caffe
