#include <vector>
#include "caffe/layers/proposal_layer.hpp"

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ProposalParameter& param = this->layer_param_.proposal_param();

  feat_stride_ = param.feat_stride();
  anchor_base_size_ = param.anchor_base_size();
  if (param.anchor_scale() == 3)
  {
	  anchor_scale_.push_back(8.0);
	  anchor_scale_.push_back(16.0);
	  anchor_scale_.push_back(32.0);
  }
  else
  {
	  anchor_scale_.push_back(32.0);
  }
  if (param.anchor_ratio() == 3)
  {
	  anchor_ratio_.push_back(0.5);
	  anchor_ratio_.push_back(1.0);
	  anchor_ratio_.push_back(2.0);
  }
  else
  {
	  anchor_ratio_.push_back(1.0);
  }
  max_rois_ = param.max_rois();
  rpn_min_size_ = param.rpn_min_size();
  pre_nms_topn_ = param.pre_nms_topn();
  rpn_nms_thresh_ = param.rpn_nms_thresh();
  Generate_anchors();
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
  if(top.size() > 1)
  {
    vector<int> top_shape1(2, 1);
    top_shape1.push_back(1);
    top_shape1.push_back(1);
    top[1]->Reshape(top_shape1);
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* score = bottom[0]->cpu_data(); // data order [n,c,h,w]
	const Dtype* bbox_deltas = bottom[1]->cpu_data(); // data order [n,c,h,w]
	const Dtype* im_info = bottom[2]->cpu_data(); // data order [h,w,c]
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	//float thresh = 0.3; //
	vector<vector<float> > select_anchor;
	vector<float> confidence;
	vector<vector<float> > bbox;
	int anchor_num = anchor_scale_.size()*anchor_ratio_.size();

	// TODO: stored data order is different from python version, may need adjustment
	for (int k = 0; k < anchor_num; k++)
	{
		float w = anchor_boxes_[4 * k + 2] - anchor_boxes_[4 * k] + 1;
		float h = anchor_boxes_[4 * k + 3] - anchor_boxes_[4 * k + 1] + 1;
		float x_ctr = anchor_boxes_[4 * k] + 0.5 * w;
		float y_ctr = anchor_boxes_[4 * k + 1] + 0.5 * h;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				//if (score[anchor_num*height*width + (k * height + i) * width + j] >= thresh)
				{
					vector<float> tmp_anchor;
					vector<float> tmp_bbox;
					tmp_anchor.push_back(j * feat_stride_+ x_ctr);
					tmp_anchor.push_back(i * feat_stride_+ y_ctr);
					tmp_anchor.push_back(w);
					tmp_anchor.push_back(h);
					//LOG(INFO)<<j * feat_stride_+ x_ctr<<" "<<i * feat_stride_+ y_ctr<<" "<<w<<" "<<h<<std::endl;
					select_anchor.push_back(tmp_anchor);
					confidence.push_back(score[anchor_num*height*width + (k * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[(4 * k * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[((4 * k +1) * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[((4 * k + 2) * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[((4 * k + 3) * height + i) * width + j]);
					bbox.push_back(tmp_bbox);
				}
			}
		}
	}

	vector<vector<float> > pred_boxes;
	bbox_transform_inv(im_info[1], im_info[0], bbox, select_anchor, pred_boxes);

	float min_size = rpn_min_size_ * im_info[2];
	filter_boxes(pred_boxes, confidence, min_size);

	int count = pred_boxes.size();
	std::vector<std::pair<Dtype, int> > score_index_pair(count);
	for (int j = 0; j < count; ++j) {
	  score_index_pair[j] = std::make_pair(confidence[j], j);
	}

	count = count < pre_nms_topn_? count:pre_nms_topn_;
    std::partial_sort(score_index_pair.begin(), score_index_pair.begin() + count,
        score_index_pair.end(), std::greater<std::pair<Dtype, int> >());

    vector<int> indices;
    applynmsfast(pred_boxes, score_index_pair, rpn_nms_thresh_, max_rois_, indices);

    //apply_nms(pred_boxes, confidence);
	//int num = pred_boxes.size() > max_rois_ ? max_rois_ : pred_boxes.size();
    int num = indices.size();

	vector<int> proposal_shape;
	proposal_shape.push_back(num);
	proposal_shape.push_back(5);
	top[0]->Reshape(proposal_shape);
	Dtype* top_data = top[0]->mutable_cpu_data();
	for (int i = 0; i < num; i++)
	{
	    int index = indices[i];
		top_data[5 * i] = 0; // batch
		top_data[5 * i + 1] = pred_boxes[index][0];
		top_data[5 * i + 2] = pred_boxes[index][1];
		top_data[5 * i + 3] = pred_boxes[index][2];
		top_data[5 * i + 4] = pred_boxes[index][3];
	}

	if(top.size() > 1)
	{
	    vector<int> score_shape;
	    score_shape.push_back(num);
	    score_shape.push_back(1);
	    top[1]->Reshape(score_shape);
	    Dtype* top_data1 = top[1]->mutable_cpu_data();
	    for (int i = 0; i < num; i++)
	    {
	        int index = indices[i];
	        top_data1[i] = confidence[index];
	    }
	}
}


//generate anchors
template <typename Dtype>
void ProposalLayer<Dtype>::Generate_anchors() {
	vector<float> base_anchor;
	base_anchor.push_back(0);
	base_anchor.push_back(0);
	base_anchor.push_back(anchor_base_size_ - 1);
	base_anchor.push_back(anchor_base_size_ - 1);
	vector<float> anchors_ratio;
	_ratio_enum(base_anchor, anchors_ratio);
	_scale_enum(anchors_ratio, anchor_boxes_);
}

template <typename Dtype>
void ProposalLayer<Dtype>::_whctrs(vector <float> anchor, vector<float> &ctrs) {
	float w = anchor[2] - anchor[0] + 1;
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);
	ctrs.push_back(w);
	ctrs.push_back(h);
	ctrs.push_back(x_ctr);
	ctrs.push_back(y_ctr);
}

template <typename Dtype>
void ProposalLayer<Dtype>::_ratio_enum(vector<float> anchor, vector<float> &anchors_ratio) {
	vector<float> ctrs;
	_whctrs(anchor, ctrs);
	float size = ctrs[0] * ctrs[1];
	int ratio_num = anchor_ratio_.size();
	for (int i = 0; i < ratio_num; i++)
	{
		float ratio = size / anchor_ratio_[i];
		int ws = int(round(sqrt(ratio)));
		int hs = int(round(ws * anchor_ratio_[i]));
		vector<float> ctrs_in;
		ctrs_in.push_back(ws);
		ctrs_in.push_back(hs);
		ctrs_in.push_back(ctrs[2]);
		ctrs_in.push_back(ctrs[3]);
		_mkanchors(ctrs_in, anchors_ratio);
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::_scale_enum(vector<float> anchors_ratio, vector<float> &anchor_boxes) {
	int anchors_ratio_num = anchors_ratio.size() / 4;
	for (int i = 0; i < anchors_ratio_num; i++)
	{
		vector<float> anchor;
		anchor.push_back(anchors_ratio[i * 4]);
		anchor.push_back(anchors_ratio[i * 4 + 1]);
		anchor.push_back(anchors_ratio[i * 4 + 2]);
		anchor.push_back(anchors_ratio[i * 4 + 3]);
		vector<float> ctrs;
		_whctrs(anchor, ctrs);
		int scale_num = anchor_scale_.size();
		for (int j = 0; j < scale_num; j++)
		{
			float ws = ctrs[0] * anchor_scale_[j];
			float hs = ctrs[1] * anchor_scale_[j];
			vector<float> ctrs_in;
			ctrs_in.push_back(ws);
			ctrs_in.push_back(hs);
			ctrs_in.push_back(ctrs[2]);
			ctrs_in.push_back(ctrs[3]);
			_mkanchors(ctrs_in, anchor_boxes);
		}
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::_mkanchors(vector<float> ctrs, vector<float> &anchors) {
	anchors.push_back(ctrs[2] - 0.5*(ctrs[0] - 1));
	anchors.push_back(ctrs[3] - 0.5*(ctrs[1] - 1));
	anchors.push_back(ctrs[2] + 0.5*(ctrs[0] - 1));
	anchors.push_back(ctrs[3] + 0.5*(ctrs[1] - 1));
}

template <typename Dtype>
void ProposalLayer<Dtype>::bbox_transform_inv(int img_width, int img_height, vector<vector<float> > bbox, vector<vector<float> > select_anchor, vector<vector<float> > &pred)
{
	int num = bbox.size();
	for (int i = 0; i< num; i++)
	{
			float dx = bbox[i][0];
			float dy = bbox[i][1];
			float dw = bbox[i][2];
			float dh = bbox[i][3];
			float pred_ctr_x = select_anchor[i][0] + select_anchor[i][2]*dx;
			float pred_ctr_y = select_anchor[i][1] + select_anchor[i][3] *dy;
			float pred_w = select_anchor[i][2] * exp(dw);
			float pred_h = select_anchor[i][3] * exp(dh);
	        //LOG(INFO)<<"w="<<pred_w<<", h="<<pred_h<<", x="<<pred_ctr_x<<", y="<<pred_ctr_y<<std::endl;
			vector<float> tmp_pred;
			tmp_pred.push_back(max(min(pred_ctr_x - 0.5* pred_w, img_width - 1), 0));
			tmp_pred.push_back(max(min(pred_ctr_y - 0.5* pred_h, img_height - 1), 0));
			tmp_pred.push_back(max(min(pred_ctr_x + 0.5* pred_w, img_width - 1), 0));
			tmp_pred.push_back(max(min(pred_ctr_y + 0.5* pred_h, img_height - 1), 0));
			pred.push_back(tmp_pred);
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::filter_boxes(vector<vector<float> > &pred_boxes, vector<float> &confidence, float min_size)
{
  for (int i = 0; i < pred_boxes.size()-1; i++)
  {
    float ws = pred_boxes[i][2] - pred_boxes[i][0] + 1;
    float hs = pred_boxes[i][3] - pred_boxes[i][1] + 1;
    bool keep = (ws >= min_size) && (hs >= min_size);
    if(!keep)
    {
      pred_boxes.erase(pred_boxes.begin() + i);
      confidence.erase(confidence.begin() + i);
      i--;
    }
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::applynmsfast(vector<vector<float> > &pred_boxes, vector<pair<Dtype, int> > &score_index_vec,
    const float nms_threshold, const int top_k, vector<int> &indices) {
  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices.clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    float x1 = pred_boxes[idx][0];
    float y1 = pred_boxes[idx][1];
    float x2 = pred_boxes[idx][2];
    float y2 = pred_boxes[idx][3];
    float areas = (x2 - x1 + 1) * (y2 - y1 + 1);
    bool keep = true;
    for (int k = 0; k < indices.size(); ++k) {
      if (keep) {
        const int kept_idx = indices[k];
        float x11 = pred_boxes[kept_idx][0];
        float y11 = pred_boxes[kept_idx][1];
        float x21 = pred_boxes[kept_idx][2];
        float y21 = pred_boxes[kept_idx][3];
        float areas1 = (x21 - x11 + 1) * (y21 - y11 + 1);

        const Dtype inter_xmin = max(x1, x11);
        const Dtype inter_ymin = max(y1, y11);
        const Dtype inter_xmax = min(x2, x21);
        const Dtype inter_ymax = min(y2, y21);
        const Dtype inter_width = max(inter_xmax - inter_xmin + 1, 0);
        const Dtype inter_height = max(inter_ymax - inter_ymin + 1, 0);
        const Dtype inter_size = inter_width * inter_height;

        float overlap = inter_size / (areas + areas1 - inter_size);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices.push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
  }
  if(indices.size()>top_k)
    indices.resize(top_k);
}

/*
template <typename Dtype>
void ProposalLayer<Dtype>::apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence)
{
	for (int i = 0; i < pred_boxes.size()-1; i++)
	{
		float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1) *(pred_boxes[i][3] - pred_boxes[i][1] + 1);
		for (int j = i + 1; j < pred_boxes.size(); j++)
		{
			float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1) *(pred_boxes[j][3] - pred_boxes[j][1] + 1);

			float x1 = max(pred_boxes[i][0], pred_boxes[j][0]);
			float y1 = max(pred_boxes[i][1], pred_boxes[j][1]);
			float x2 = min(pred_boxes[i][2], pred_boxes[j][2]);
			float y2 = min(pred_boxes[i][3], pred_boxes[j][3]);

			float width = x2 - x1;
			float height = y2 - y1;
			if (width > 0 && height > 0)
			{
				float IOU = width * height / (s1 + s2 - width * height);
				if (IOU > 0.7) //
				{
					if (confidence[i] >= confidence[j])
					{
						pred_boxes.erase(pred_boxes.begin() + j);
						confidence.erase(confidence.begin() + j);
						j--;
					}
					else
					{
						pred_boxes.erase(pred_boxes.begin() + i);
						confidence.erase(confidence.begin() + i);
						i--;
						break;
					}
				}
			}
		}
	}
}
*/

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe
