#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <queue>
#define HelperMin(a, b) std::min(a, b)
#define HelperMax(a, b) std::max(a, b)

#include "caffe/layers/non_max_suppression_layer.hpp"

namespace caffe {

template <typename Dtype>
void NonMaxSuppressionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const NonMaxSuppressionParameter& non_max_suppression_param = this->layer_param_.non_max_suppression_param();
  max_output_boxes_per_class_ = non_max_suppression_param.max_output_boxes_per_class();
  iou_threshold_ =  non_max_suppression_param.iou_threshold();
  score_threshold_ = non_max_suppression_param.score_threshold();
  center_point_box_ = non_max_suppression_param.center_point_box();

  CHECK_EQ(bottom[0]->num_axes(), 3) << "bottom[0] must have 3 axes.";
  CHECK_EQ(bottom[1]->num_axes(), 3) << "bottom[1] must have 3 axes.";

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "The boxes and scores should have same num_batches.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(2)) << "The boxes and scores should have same spatial_dimension.";
  CHECK_EQ(bottom[0]->shape(2), 4) << "This coordinates axis of boxes must have shape 4.";
  num_batches_ = bottom[0]->shape(0);
  num_classes_ = bottom[1]->shape(1);
  num_boxes_ = bottom[0]->shape(1);

  CHECK_GE(iou_threshold_, 0) << "The iou_threshold must not be less than 0.";
  CHECK_LE(iou_threshold_, 1) << "The iou_threshold must not be greater than 1.";

}

template <typename Dtype>
void NonMaxSuppressionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> shape(2);
  shape[0] = max_output_boxes_per_class_ * num_classes_ * num_batches_;
  shape[1] = 3;
  top[0]->Reshape(shape);
}


template <typename Dtype>
void NonMaxSuppressionLayer<Dtype>::MaxMin(float lhs, float rhs, float& min, float& max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}


template <typename Dtype>
bool NonMaxSuppressionLayer<Dtype>::SuppressByIOU(const Dtype* boxes_data, int64_t box_index1, int64_t box_index2,
                          int64_t center_point_box, float iou_threshold) {
  float x1_min{};
  float y1_min{};
  float x1_max{};
  float y1_max{};
  float x2_min{};
  float y2_min{};
  float x2_max{};
  float y2_max{};

  const Dtype* box1 = boxes_data + 4 * box_index1;
  const Dtype* box2 = boxes_data + 4 * box_index2;
  // center_point_box_ only support 0 or 1
  if (0 == center_point_box) {
    // boxes data format [y1, x1, y2, x2],
    MaxMin(box1[1], box1[3], x1_min, x1_max);
    MaxMin(box1[0], box1[2], y1_min, y1_max);
    MaxMin(box2[1], box2[3], x2_min, x2_max);
    MaxMin(box2[0], box2[2], y2_min, y2_max);
  } else {
    // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
    float box1_width_half = box1[2] / 2;
    float box1_height_half = box1[3] / 2;
    float box2_width_half = box2[2] / 2;
    float box2_height_half = box2[3] / 2;

    x1_min = box1[0] - box1_width_half;
    x1_max = box1[0] + box1_width_half;
    y1_min = box1[1] - box1_height_half;
    y1_max = box1[1] + box1_height_half;

    x2_min = box2[0] - box2_width_half;
    x2_max = box2[0] + box2_width_half;
    y2_min = box2[1] - box2_height_half;
    y2_max = box2[1] + box2_height_half;
  }

  const float intersection_x_min = HelperMax(x1_min, x2_min);
  const float intersection_y_min = HelperMax(y1_min, y2_min);
  const float intersection_x_max = HelperMin(x1_max, x2_max);
  const float intersection_y_max = HelperMin(y1_max, y2_max);

  const float intersection_area = HelperMax(intersection_x_max - intersection_x_min, .0f) *
                                  HelperMax(intersection_y_max - intersection_y_min, .0f);

  if (intersection_area <= .0f) {
    return false;
  }

  const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
  const float union_area = area1 + area2 - intersection_area;

  if (area1 <= .0f || area2 <= .0f || union_area <= .0f) {
    return false;
  }

  const float intersection_over_union = intersection_area / union_area;

  return intersection_over_union > iou_threshold;
}

template <typename Dtype>
void NonMaxSuppressionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* boxes_data = bottom[0]->cpu_data();
  const Dtype* scores_data = bottom[1]->cpu_data();

  std::vector<int64_t> selected_indices;
   for (int64_t batch_index = 0; batch_index < num_batches_; ++batch_index) {
     for (int64_t class_index = 0; class_index < num_classes_; ++class_index) {
       int64_t box_score_offset = (batch_index * num_classes_ + class_index) * num_boxes_;
       int64_t box_offset = batch_index * num_boxes_ * 4;

       // Filter by score_threshold_
       std::priority_queue<ScoreIndexPair, std::deque<ScoreIndexPair>> sorted_scores_with_index;
       const Dtype* class_scores = scores_data + box_score_offset;
       for (int64_t box_index = 0; box_index < num_boxes_; ++box_index, ++class_scores) {
         if (*class_scores > score_threshold_) {
           sorted_scores_with_index.push(ScoreIndexPair(*class_scores, box_index));
         }
       }

       ScoreIndexPair next_top_score;
       std::vector<int64_t> selected_indices_inside_class;
       // Get the next box with top score, filter by iou_threshold
       while (!sorted_scores_with_index.empty()) {
         next_top_score = sorted_scores_with_index.top();
         sorted_scores_with_index.pop();

         bool selected = true;
         // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
         for (int64_t selected_index : selected_indices_inside_class) {
           if (SuppressByIOU(boxes_data + box_offset, selected_index, next_top_score.index_,
                             center_point_box_, iou_threshold_)) {
             selected = false;
             break;
           }
         }

         if (selected) {
           if (max_output_boxes_per_class_ > 0 &&
               static_cast<int64_t>(selected_indices_inside_class.size()) >= max_output_boxes_per_class_) {
             break;
           }
           selected_indices_inside_class.push_back(next_top_score.index_);
           //selected_indices.emplace_back(batch_index, class_index, next_top_score.index_);
           selected_indices.emplace_back(batch_index);
           selected_indices.emplace_back(class_index);
           selected_indices.emplace_back(next_top_score.index_);
         }
       }  //while

     }    //for class_index
   }      //for batch_index

   Dtype* top_data = top[0]->mutable_cpu_data();
   const int64_t num_selected = selected_indices.size();
   for(int i=0; i<num_selected; i++)
     top_data[i] = selected_indices[i];
   // Note: fixed output shape might be larger than the count of selected indices,
   // the following parts will be filled by 0.
}

INSTANTIATE_CLASS(NonMaxSuppressionLayer);
REGISTER_LAYER_CLASS(NonMaxSuppression);

}  // namespace caffe
