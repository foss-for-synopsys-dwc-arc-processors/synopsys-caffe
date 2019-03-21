#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/nms_gather_layer.hpp"

namespace caffe {

template <typename Dtype>
void NMSGatherLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const NMSGatherParameter& nms_gather_param = this->layer_param_.nms_gather_param();
  top_k_ = nms_gather_param.top_k();
  iou_threshold_ =  nms_gather_param.iou_threshold();
  axis_ = bottom[0]->CanonicalAxisIndex(nms_gather_param.axis());
  CHECK_EQ(bottom[0]->num_axes(), 2) << "bottom[0] must have 2 axes.";
  CHECK_EQ(bottom[0]->shape(1-axis_), 4) << "Coordinates axis must have shape 4.";
  CHECK_GE(top_k_, 1) << "top k must not be less than 1.";
  CHECK_GE(axis_, 0) << "axis must not be less than 0.";
  CHECK_LE(axis_, bottom[0]->num_axes()) <<
    "axis must be less than or equal to the number of axis.";
  CHECK_LE(top_k_, bottom[0]->shape(axis_))
    << "top_k must be less than or equal to the dimension of the axis.";
}

template <typename Dtype>
void NMSGatherLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  CHECK_GE(num_axes, 1) << "the dimension of input should be larger than or equal to 1";
  const NMSGatherParameter& nms_gather_param = this->layer_param_.nms_gather_param();
  gather_axis_ = bottom[0]->CanonicalAxisIndex(nms_gather_param.axis());
  indices_shape_.clear();
  indices_shape_.push_back(top_k_); //only 1 dimension for topk selection

  if (indices_shape_.size() == 1 && indices_shape_[0] == 0) {
 	indices_dim_ = 0;
  }
  else {
	indices_dim_ = indices_shape_.size();
	int count = 1;
	for (int i = 0; i < indices_shape_.size(); ++i) {
	  count *= indices_shape_[i];
	}
  }

  // Initialize with the first blob
  // The result shape is params.shape[-1:axis] + indices.shape +
  // params.shape[axis + 0:].
  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(bottom_shape.size() + indices_dim_ - 1);
  num_gather_ = bottom[0]->count(0, gather_axis_);
  gather_size_ = bottom[0]->count(gather_axis_ + 1);

  for (int i = 0; i < gather_axis_; ++i) {
	top_shape[i] = bottom_shape[i];
  }
  for (int i = 0; i < indices_dim_; ++i) {
	top_shape[i + gather_axis_] = indices_shape_[i];
  }
  for (int i = gather_axis_ + 1; i < num_axes; ++i) {
	top_shape[i + indices_dim_ - 1] = bottom_shape[i];
  }
  top[0]->Reshape(top_shape);
}


template <typename Dtype>
void NMSGatherLayer<Dtype>::apply_nms(vector<vector<Dtype> > &pred_boxes, vector<int> &indices, float iou_threshold)
{
	for (int i = 0; i < indices.size()-1; i++)
	{
		float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1) *(pred_boxes[i][3] - pred_boxes[i][1] + 1);
		for (int j = i + 1; j < indices.size(); j++)
		{
			float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1) *(pred_boxes[j][3] - pred_boxes[j][1] + 1);
			// Bounding boxes are supplied as [y1, x1, y2, x2] in TensorFlow
			float y1 = std::max(pred_boxes[i][0], pred_boxes[j][0]);
			float x1 = std::max(pred_boxes[i][1], pred_boxes[j][1]);
			float y2 = std::min(pred_boxes[i][2], pred_boxes[j][2]);
			float x2 = std::min(pred_boxes[i][3], pred_boxes[j][3]);

			float width = x2 - x1;
			float height = y2 - y1;
			if (width > 0 && height > 0)
			{
				// intersection-over-union (IOU) overlap
				float IOU = width * height / (s1 + s2 - width * height);
				if (IOU > iou_threshold)
				{
					indices.erase(indices.begin() + j);
					j--; //erase make the indices count decrease
				}
			}
		}
	}
}


template <typename Dtype>
void NMSGatherLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  indices_.clear();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  for (int j = 0; j < bottom[0]->shape(axis_); ++j) {
	  indices_.push_back(j); //initialize indices_
  }
  vector<vector<Dtype> > pred_boxes; //reorder bottom[0] to [num, 4] format
  if (axis_ == 0) {
	for (int i = 0; i < bottom[0]->shape(0); ++i) {
	  vector<Dtype> tmp_bbox;
	  for (int j = 0; j < 4; ++j){
	    tmp_bbox.push_back(bottom_data[i*4+j]);
	  }
	  pred_boxes.push_back(tmp_bbox);
	  }
  } else { //axis_ == 1
	for (int j = 0; j < bottom[0]->shape(1); ++j){
	  vector<Dtype> tmp_bbox;
	  for (int i = 0; i < 4; ++i){
		tmp_bbox.push_back(bottom_data[i*bottom[0]->shape(1)+j]);
	  }
      pred_boxes.push_back(tmp_bbox);
	}
  }
  apply_nms(pred_boxes, indices_, iou_threshold_);

  vector<int> bottom_shape = bottom[0]->shape();
  const Dtype* bottom_data_c = bottom[0]->cpu_data(); //reset pointer
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int bottom_gather_axis = bottom[0]->shape(gather_axis_);
  int num_c = 0;
  for (int m = 0; m < num_gather_; ++m) {
	for (int n = 0; n < top_k_; ++n) {
	  const int top_offset = num_c * gather_size_;
      const int bottom_offset =
		  (m * bottom_gather_axis + indices_[n]) * gather_size_;
      caffe_copy(gather_size_,
		  bottom_data_c + bottom_offset, top_data + top_offset);
      num_c += 1;
	}
  }
}

INSTANTIATE_CLASS(NMSGatherLayer);
REGISTER_LAYER_CLASS(NMSGather);

}  // namespace caffe
