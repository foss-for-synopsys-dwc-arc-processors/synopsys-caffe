// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Synopsys Inc - Backward_cpu implementation
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int count = bottom[0]->count();
  caffe_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.cpu_data();
  const int batch_size = bottom[0]->num();
  // Number of ROIs
  int num_rois = top[0]->num();

  // (n, c, h, w) coords in bottom data
  for (int n = 0; n < batch_size; ++n) {
	  Dtype* offset_bottom_diff = bottom_diff;

	  for (int c = 0; c < channels_; ++c) {

		  for (int h = 0; h < height_; ++h) {
			  for (int w = 0; w < width_; ++w) {

				  const Dtype* offset_top_diff = top_diff + top[0]->offset(0, c);
				  const int* offset_argmax_data = argmax_data + max_idx_.offset(0, c);
				  const Dtype* offset_bottom_rois = bottom_rois;

				  // Accumulate gradient over all ROIs that pooled this element
				  for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
					  int roi_batch_ind = offset_bottom_rois[0];
					  CHECK_GE(roi_batch_ind, 0);
					  CHECK_LT(roi_batch_ind, batch_size);

					  // Skip if ROI's batch index doesn't match n
					  if (n != roi_batch_ind) {
						  // Increment ROI data pointer
						  offset_bottom_rois += bottom[1]->offset(1);
						  offset_top_diff += top[0]->offset(1);
						  offset_argmax_data += max_idx_.offset(1);
						  continue;
					  }

					  int roi_start_w = round(
							  offset_bottom_rois[1] * spatial_scale_);
					  int roi_start_h = round(
							  offset_bottom_rois[2] * spatial_scale_);
					  int roi_end_w = round(
							  offset_bottom_rois[3] * spatial_scale_);
					  int roi_end_h = round(
							  offset_bottom_rois[4] * spatial_scale_);


					  // Skip if ROI doesn't include (h, w)
					  const bool in_roi = (w >= roi_start_w && w <= roi_end_w
							  && h >= roi_start_h && h <= roi_end_h);
					  if (!in_roi) {
						  // Increment ROI data pointer
						  offset_bottom_rois += bottom[1]->offset(1);
						  offset_top_diff += top[0]->offset(1);
						  offset_argmax_data += max_idx_.offset(1);
						  continue;
					  }

					  // Force malformed ROIs to be 1x1
					  int roi_width = max(roi_end_w - roi_start_w + 1, 1);
					  int roi_height = max(roi_end_h - roi_start_h + 1, 1);

					  Dtype bin_size_h = static_cast<Dtype>(roi_height)
										/ static_cast<Dtype>(pooled_height_);
					  Dtype bin_size_w = static_cast<Dtype>(roi_width)
										/ static_cast<Dtype>(pooled_width_);

					  // Compute feasible set of pooled units that could have pooled
					  // this bottom unit

					  int phstart = floor(static_cast<Dtype>(h - roi_start_h)
							  / bin_size_h);
					  int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1)
							  / bin_size_h);
					  int pwstart = floor(static_cast<Dtype>(w - roi_start_w)
							  / bin_size_w);
					  int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1)
							  / bin_size_w);

					  phstart = min(max(phstart, 0), pooled_height_);
					  phend = min(max(phend, 0), pooled_height_);
					  pwstart = min(max(pwstart, 0), pooled_width_);
					  pwend = min(max(pwend, 0), pooled_width_);

					  for (int ph = phstart; ph < phend; ++ph) {
						  for (int pw = pwstart; pw < pwend; ++pw) {
							  if (offset_argmax_data[ph * pooled_width_ + pw] == (h * width_ + w)) {
								  offset_bottom_diff[h * width_ + w] += offset_top_diff[ph * pooled_width_ + pw];
							  }
						  }
					  }
					  // Increment ROI data pointer
					  offset_bottom_rois += bottom[1]->offset(1);
					  offset_top_diff += top[0]->offset(1);
					  offset_argmax_data += max_idx_.offset(1);
				  }
			  }
		  }
		  // Increment data pointers by one channel
		  offset_bottom_diff += bottom[0]->offset(0, 1);
	  }
	  bottom_diff += bottom[0]->offset(1);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe
