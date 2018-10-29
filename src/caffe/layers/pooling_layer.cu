#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define SATURATE_MAX 4095
#define SATURATE_MIN -4096

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
	//const int pad_h, const int pad_w,
	const int pad_top, const int pad_left, const int pad_bottom, const int pad_right, //CUSTOMIZATION
	Dtype* const top_data, const int output_shift_instead_division, const bool saturate) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    //<--CUSTOMIZATION
    //int hstart = ph * stride_h - pad_h;
    //int wstart = pw * stride_w - pad_w;
    int hstart = ph * stride_h - pad_top;
    int wstart = pw * stride_w - pad_left;
    //int hend = min(hstart + kernel_h, height + pad_h);
    //int wend = min(wstart + kernel_w, width + pad_w);
    int hend = min(hstart + kernel_h, height + pad_bottom);
    int wend = min(wstart + kernel_w, width + pad_right);
    //CUSTOMIZATION-->
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    if (output_shift_instead_division != Dtype(0)) {
      top_data[index] = aveval / output_shift_instead_division;
      top_data[index] = rint(top_data[index]);
      if(saturate)
      {
        if(top_data[index] > SATURATE_MAX)
          top_data[index] = SATURATE_MAX;
        if(top_data[index] < SATURATE_MIN)
          top_data[index] = SATURATE_MIN;
      }
    }
    else{
      if(saturate)
      {
    	top_data[index] = aveval;
        if(top_data[index] > SATURATE_MAX)
          top_data[index] = SATURATE_MAX;
        if(top_data[index] < SATURATE_MIN)
          top_data[index] = SATURATE_MIN;
      }
      else //original implementation
        top_data[index] = aveval / pool_size;
    }
  }
}

//<--CUSTOMIZATION
template <typename Dtype>
__global__ void AvePoolForward_TF(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
	//const int pad_h, const int pad_w,
	const int pad_top, const int pad_left, const int pad_bottom, const int pad_right, //CUSTOMI
    Dtype* const top_data, const int output_shift_instead_division, const bool saturate) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    //<--CUSTOMIZATION
    //int hstart = ph * stride_h - pad_h;
    //int wstart = pw * stride_w - pad_w;
    int hstart = ph * stride_h - pad_top;
    int wstart = pw * stride_w - pad_left;
    //int hend = min(hstart + kernel_h, height + pad_h);
    //int wend = min(wstart + kernel_w, width + pad_w);
    int hend = min(hstart + kernel_h, height + pad_bottom);
    int wend = min(wstart + kernel_w, width + pad_right);
    //CUSTOMIZATION-->
    const int full_pool_size = (hend - hstart) * (wend - wstart); //
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    const int pool_size = (hend - hstart) * (wend - wstart); //
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    if (output_shift_instead_division != Dtype(0)) {
      if (full_pool_size == pool_size)
        top_data[index] = aveval / output_shift_instead_division;
      else {
    	//special fix: Non zero paddings for the case when:
    	//1)the kernel runs off the edge only by 1 pixel
    	//2)and the kernel_size-1 is a power of 2
    	//refer to "Repair by changing padding" at
    	//https://wwwin.synopsys.com/~tpennell/cnn_papers/29_average_pooling_repair_shop.htm
    	bool wfix = (pw * stride_w - pad_left == -1) || (wstart + kernel_w - width == 1);
    	bool hfix = (ph * stride_h - pad_top == -1) || (hstart + kernel_h - height == 1);

        if (wfix && hfix)
        {
		  Dtype aveval_fix;
		  for (int h = hstart; h < hend; ++h) {
			aveval_fix = 0;
			for (int w = wstart; w < wend; ++w) {
		      aveval_fix += bottom_slice[h * width + w];
			}
			aveval += rint(aveval_fix / (wend - wstart));
		  }

		  for (int w = wstart; w < wend; ++w) {
			aveval_fix = 0;
			for (int h = hstart; h < hend; ++h) {
			  aveval_fix += bottom_slice[h * width + w];
			}
			aveval += rint(aveval_fix / (hend - hstart));
		  }

		  aveval_fix = 0;
		  for (int w = wstart; w < wend; ++w) {
			Dtype aveval_fix_tmp = 0;
		    for (int h = hstart; h < hend; ++h) {
		      aveval_fix_tmp += bottom_slice[h * width + w];
			}
		    aveval_fix += rint(aveval_fix_tmp / (hend - hstart));
		  }
		  aveval += rint(aveval_fix / (wend - wstart));

		  top_data[index] = aveval / output_shift_instead_division;
    	}

    	else if (hfix && !wfix)
    	{
		  Dtype aveval_fix;
		  for (int w = wstart; w < wend; ++w) {
			aveval_fix = 0;
			for (int h = hstart; h < hend; ++h) {
			  aveval_fix += bottom_slice[h * width + w];
			}
			aveval += rint(aveval_fix / (hend - hstart));
		  }
		  top_data[index] = aveval / output_shift_instead_division;
    	}

    	else if (wfix && !hfix)
    	{
		  Dtype aveval_fix;
		  for (int h = hstart; h < hend; ++h) {
			aveval_fix = 0;
			for (int w = wstart; w < wend; ++w) {
			  aveval_fix += bottom_slice[h * width + w];
			}
			aveval += rint(aveval_fix / (wend - wstart));
		  }
		  top_data[index] = aveval / output_shift_instead_division;
    	}

    	else
          top_data[index] = aveval / output_shift_instead_division * full_pool_size / pool_size;
      }
      top_data[index] = rint(top_data[index]);
      if(saturate)
      {
        if(top_data[index] > SATURATE_MAX)
          top_data[index] = SATURATE_MAX;
        if(top_data[index] < SATURATE_MIN)
          top_data[index] = SATURATE_MIN;
      }
    }

    else{
      if(saturate)
      {
      	top_data[index] = aveval;
        if(top_data[index] > SATURATE_MAX)
          top_data[index] = SATURATE_MAX;
        if(top_data[index] < SATURATE_MIN)
          top_data[index] = SATURATE_MIN;
      }
      else //original implementation
        top_data[index] = aveval / pool_size;
    }
  }
}
//CUSTOMIZATION-->

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = 0.;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  //<--CUSOMIZATION
  int pad_top=0, pad_bottom=0, pad_left=0, pad_right=0;
  switch (pad_type_) {
    case 0:
	  if (pad_l_ != 0 || pad_r_ != 0 || pad_t_ != 0 || pad_b_ != 0) {
		pad_top = pad_t_;
		pad_bottom = pad_b_;
		pad_left = pad_l_;
		pad_right = pad_r_;
	  } else {
		pad_top = pad_h_;
		pad_bottom = pad_h_;
		pad_left = pad_w_;
		pad_right = pad_w_;
	  }
      break;
    case 1:  //for "SAME"padding
      int pad_along_height, pad_along_width;
      if (height_ % stride_h_ == 0)
        pad_along_height = (kernel_h_ - stride_h_)>0 ? (kernel_h_ - stride_h_) : 0;
      else
        pad_along_height = (kernel_h_ - height_ % stride_h_)>0 ? (kernel_h_ - height_ % stride_h_) : 0;
      if (width_ % stride_w_ == 0)
        pad_along_width = (kernel_w_ - stride_w_)>0 ? (kernel_w_ - stride_w_) : 0;
      else
        pad_along_width = (kernel_w_ - width_ % stride_w_)>0 ? (kernel_w_ - width_ % stride_w_): 0;
      pad_top = pad_along_height / 2;
      pad_bottom = pad_along_height - pad_top;
      pad_left = pad_along_width / 2;
      pad_right = pad_along_width - pad_left;
      break;
    default:
      LOG(FATAL) << "Unknown pooling padding type.";
      break;
  }
  //CUSTOMIZATION-->

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_,
		//pad_h_, pad_w_,
		pad_top, pad_left, //CUSTOMIZATION
		top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_,
		//pad_h_, pad_w_,
		pad_top, pad_left, pad_bottom, pad_right, //CUSTOMIZATION
		top_data, output_shift_instead_division_, saturate_);
    break;
  //<--CUSTOMIZATION
  case PoolingParameter_PoolMethod_AVE_EXC_PAD:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward_TF<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_,
		//pad_h_, pad_w_,
		pad_top, pad_left, pad_bottom, pad_right, //CUSTOMIZATION
		top_data, output_shift_instead_division_, saturate_);
    break;
    //CUSTOMIZATION-->
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
		  top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
	//const int pad_h, const int pad_w,
	const int pad_top, const int pad_left, const int pad_bottom, const int pad_right, //CUSTOMIZATION
	Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    //<--CUSTOMIZATION
    //const int phstart =
    //     (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    //const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    //const int pwstart =
    //     (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    //const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    const int phstart =
         (h + pad_top < kernel_h) ? 0 : (h + pad_top - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_bottom) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_left < kernel_w) ? 0 : (w + pad_left - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_right) / stride_w + 1, pooled_width);
    //CUSTOMIZATION-->
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
	//const int pad_h, const int pad_w,
	const int pad_top, const int pad_left, const int pad_bottom, const int pad_right, //CUSTOMIZATION
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    //const int w = index % width + pad_w;
    //const int h = (index / width) % height + pad_h;
    const int w = index % width; //CUSTOMIZATION
    const int h = (index / width) % height; //CUSTOMIZATION
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    //<--CUSTOMIZATION
    //const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    //const int phend = min(h / stride_h + 1, pooled_height);
    //const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    //const int pwend = min(w / stride_w + 1, pooled_width);
    const int phstart = ( (h+pad_top) < kernel_h) ? 0 : ( (h+pad_top) - kernel_h) / stride_h + 1;
    const int phend = min( (h+pad_bottom) / stride_h + 1, pooled_height);
    const int pwstart = ( (w+pad_left) < kernel_w) ? 0 : ( (w+pad_left) - kernel_w) / stride_w + 1;
    const int pwend = min( (w+pad_right) / stride_w + 1, pooled_width);
    //CUSTOMIZATION-->
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
    	//<--CUSTOMIZATION
        //int hstart = ph * stride_h - pad_h;
        //int wstart = pw * stride_w - pad_w;
        //int hend = min(hstart + kernel_h, height + pad_h);
        //int wend = min(wstart + kernel_w, width + pad_w);
    	int hstart = ph * stride_h - pad_top;
    	int wstart = pw * stride_w - pad_left;
    	int hend = min(hstart + kernel_h, height + pad_bottom);
    	int wend = min(wstart + kernel_w, width + pad_right);
        //-->CUSTOMIZATION
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

//<--CUSTOMIZATION
template <typename Dtype>
__global__ void AvePoolBackward_TF(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
	//const int pad_h, const int pad_w,
	const int pad_top, const int pad_left, const int pad_bottom, const int pad_right, //CUSTOMIZATION
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
	//const int w = index % width + pad_w;
	//const int h = (index / width) % height + pad_h;
	const int w = index % width; //CUSTOMIZATION
	const int h = (index / width) % height; //CUSTOMIZATION
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    //<--CUSTOMIZATION
    //const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    //const int phend = min(h / stride_h + 1, pooled_height);
    //const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    //const int pwend = min(w / stride_w + 1, pooled_width);
    const int phstart = ( (h+pad_top) < kernel_h) ? 0 : ( (h+pad_top) - kernel_h) / stride_h + 1;
    const int phend = min( (h+pad_bottom) / stride_h + 1, pooled_height);
    const int pwstart = ( (w+pad_left) < kernel_w) ? 0 : ( (w+pad_left) - kernel_w) / stride_w + 1;
    const int pwend = min( (w+pad_right) / stride_w + 1, pooled_width);
    //CUSTOMIZATION-->
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        //<--CUSTOMIZATION
    	//int hstart = ph * stride_h - pad_h;
        //int wstart = pw * stride_w - pad_w;
        //int hend = min(hstart + kernel_h, height + pad_h);
        //int wend = min(wstart + kernel_w, width + pad_w);
      	int hstart = ph * stride_h - pad_top;
      	int wstart = pw * stride_w - pad_left;
        int hend = min(hstart + kernel_h, height + pad_bottom);
        int wend = min(wstart + kernel_w, width + pad_right);
        //-->CUSTOMIZATION
        hstart = max(hstart, 0); //
        wstart = max(wstart, 0); //
        hend = min(hend, height); //
        wend = min(wend, width); //
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}
//CUSTOMIZATION-->

template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w,
	Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  //<--CUSOMIZATION
  int pad_top=0, pad_bottom=0, pad_left=0, pad_right=0;
  switch (pad_type_) {
    case 0:
	  if (pad_l_ != 0 || pad_r_ != 0 || pad_t_ != 0 || pad_b_ != 0) {
		pad_top = pad_t_;
		pad_bottom = pad_b_;
		pad_left = pad_l_;
		pad_right = pad_r_;
	  } else {
		pad_top = pad_h_;
		pad_bottom = pad_h_;
		pad_left = pad_w_;
		pad_right = pad_w_;
	  }
      break;
    case 1:  //for "SAME"padding
      int pad_along_height, pad_along_width;
      if (height_ % stride_h_ == 0)
        pad_along_height = (kernel_h_ - stride_h_)>0 ? (kernel_h_ - stride_h_) : 0;
      else
        pad_along_height = (kernel_h_ - height_ % stride_h_)>0 ? (kernel_h_ - height_ % stride_h_) : 0;
      if (width_ % stride_w_ == 0)
        pad_along_width = (kernel_w_ - stride_w_)>0 ? (kernel_w_ - stride_w_) : 0;
      else
        pad_along_width = (kernel_w_ - width_ % stride_w_)>0 ? (kernel_w_ - width_ % stride_w_): 0;
      pad_top = pad_along_height / 2;
      pad_bottom = pad_along_height - pad_top;
      pad_left = pad_along_width / 2;
      pad_right = pad_along_width - pad_left;
      break;
    default:
      LOG(FATAL) << "Unknown pooling padding type.";
      break;
  }
  //CUSTOMIZATION-->
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
		//pad_h_, pad_w_,
		pad_top, pad_left, pad_bottom, pad_right, //CUSTOMIZATION
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_,
		//pad_h_, pad_w_,
		pad_top, pad_left, pad_bottom, pad_right, //CUSTOMIZATION
		bottom_diff);
    break;
  //<--CUSTOMIZATION
  case PoolingParameter_PoolMethod_AVE_EXC_PAD:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward_TF<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_,
		//pad_h_, pad_w_,
		pad_top, pad_left, pad_bottom, pad_right, //CUSTOMIZATION
		bottom_diff);
    break;
    //CUSTOMIZATION-->
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
