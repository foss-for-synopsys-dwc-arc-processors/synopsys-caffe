#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/nms_layer.hpp"

namespace caffe {

template <typename Dtype>
void NMSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const NMSParameter& nms_param = this->layer_param_.nms_param();
  top_k_ = nms_param.top_k();
  iou_threshold_ =  nms_param.iou_threshold();
  axis_ = bottom[0]->CanonicalAxisIndex(nms_param.axis());
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
void NMSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> shape(1);
  shape[0] = top_k_;
  top[0]->Reshape(shape);
}


template <typename Dtype>
void NMSLayer<Dtype>::apply_nms(vector<vector<Dtype> > &pred_boxes, vector<int> &indices, float iou_threshold)
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
void NMSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

  Dtype* top_data = top[0]->mutable_cpu_data();
  for(int i = 0; i < top_k_; i++)
	  top_data[i]=indices_[i];
}

INSTANTIATE_CLASS(NMSLayer);
REGISTER_LAYER_CLASS(NMS);

}  // namespace caffe
