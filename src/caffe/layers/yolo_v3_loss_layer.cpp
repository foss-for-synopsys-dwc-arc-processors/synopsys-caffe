#include "caffe/layers/yolo_v3_loss_layer.hpp"
#include <caffe/blob.hpp>
#include <float.h>
#define SEEN_NUMBER 12800

namespace caffe {

    template <typename Dtype>
    struct Box
    {
        Dtype x, y, w, h;
    };

    template <typename Dtype>
    float logistic_activate(Dtype x)
    {
        return 1. / (1 + exp(-x));
    }

    template <typename Dtype>
    float logistic_gradient(Dtype x)
    {
        return (1-x) * x;
    }

    template <typename Dtype>
    Box<Dtype> dtype_to_box(const Dtype *dtype)
    {
        Box<Dtype> box;
        box.x = dtype[0];
        box.y = dtype[1];
        box.w = dtype[2];
        box.h = dtype[3];
        return box;
    }

    template <typename Dtype>
    Box<Dtype> get_yolo_box(const Dtype *x, vector<int>& biases, int n, int SIZE, int i, int j, int net_w, int net_h, int side){
        Box<Dtype> box;
        box.x = (i + logistic_activate(x[0 * SIZE])) / side;
        box.y = (j + logistic_activate(x[1 * SIZE])) / side;
        box.w = exp(x[2 * SIZE]) * biases[2 * n] / net_w;
        box.h = exp(x[3 * SIZE]) * biases[2 * n + 1] / net_h;
        return box;
    }

    template <typename Dtype>
    Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
      Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
      Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
      return right - left;
    }

    template <typename Dtype>
    Dtype Calc_iou(const Box<Dtype> box, const Box<Dtype> truth) {
      Dtype w = Overlap(box.x, box.w, truth.x, truth.w);
      Dtype h = Overlap(box.y, box.h, truth.y, truth.h);
      if (w < 0 || h < 0) return 0;
      Dtype inter_area = w * h;
      Dtype union_area = box.w * box.h + truth.w * truth.h - inter_area;
      return inter_area / union_area;
    }

   template <typename Dtype>
    float delta_yolo_box(Box<Dtype>& truth_box, Dtype* input, 
            std::vector<int>& biases, int n, int side, Dtype* diff, 
            int index, float scale, int i, int j, int net_w, int net_h, Dtype &coord_loss, Dtype &area_loss) {
        Box<Dtype> pred_box = get_yolo_box(input + index, biases, n, side*side, i, j, net_w, net_h, side);
        float iou = Calc_iou(pred_box, truth_box);

        float tx = (truth_box.x*side - i);
        float ty = (truth_box.y*side - j);
        float tw = log((truth_box.w*net_w) / biases[2*n]);
        float th = log((truth_box.h*net_h) / biases[2*n + 1]);

        int SHIFT = side * side;
        diff[index + 0 * SHIFT] = (-1) * scale * (tx - logistic_activate(input[index + 0 * SHIFT])) * logistic_gradient(logistic_activate(input[index + 0 * SHIFT]));
        diff[index + 1 * SHIFT] = (-1) * scale * (ty - logistic_activate(input[index + 1 * SHIFT])) * logistic_gradient(logistic_activate(input[index + 1 * SHIFT]));

        diff[index + 2 * SHIFT] = (-1) * scale * (tw - input[index + 2 * SHIFT]);
        diff[index + 3 * SHIFT] = (-1) * scale * (th - input[index + 3 * SHIFT]);

        coord_loss += scale * (pow((tx - logistic_activate(input[index + 0 * SHIFT])), 2) + pow((ty - logistic_activate(input[index + 1 * SHIFT])), 2));
        area_loss += scale * (pow((tw - input[index + 2 * SHIFT]), 2) + pow((th - input[index + 3 * SHIFT]), 2));
        return iou;
    }


    template <typename Dtype>
    void YoloV3LossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);

        YoloV3LossParameter param = this->layer_param().yolo_v3_loss_param();
        const google::protobuf::RepeatedField<int> &tmpBiases_ = param.anchors();
        for(google::protobuf::RepeatedField<int>::const_iterator 
            iterator = tmpBiases_.begin(); iterator != tmpBiases_.end(); ++iterator)
        {
            biases_.push_back(*iterator);
        }
        const google::protobuf::RepeatedField<int> &tmpMasks_ = param.mask();
        for(google::protobuf::RepeatedField<int>::const_iterator 
            iterator = tmpMasks_.begin(); iterator != tmpMasks_.end(); ++iterator)
        {
            masks_.push_back(*iterator);
        }
        
        side_           = param.side();
        CHECK_GE(side_, 0) << "side size must bigger then 0";
        num_classes_    = param.num_classes();
        CHECK_GE(num_classes_, 0) << "class number must bigger then 0";
        num_boxes_      = param.num_object();
        CHECK_GE(num_boxes_, 0) << "box number must bigger then 0";
        total_num_boxes_ = param.total_object();
        seen            = 0;
        ignore_thresh_  = param.ignore_thresh();
        truth_thresh_   = param.truth_thresh();
        net_w_          = param.net_w();
        net_h_          = param.net_h();

        int input_count = bottom[0]->count(1);
        int label_count = bottom[1]->count(1);
        int tmp_input_count = side_ * side_ * (num_classes_ + (1 + 4)) * num_boxes_;
        int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);
        CHECK_EQ(input_count, tmp_input_count);
        CHECK_EQ(label_count, tmp_label_count);
    }

    template <typename Dtype>
    void YoloV3LossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::Reshape(bottom, top);
        diff_.ReshapeLike(*bottom[0]);
    }

    int int_index(vector<int>& mask, int val, int n)
    {
        int i;
        for(i = 0; i < n; ++i){
            if(mask[i] == val) return i;
        }
        return -1;
    }

    template <typename Dtype>
    void YoloV3LossLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        Dtype* input_data = bottom[0]->mutable_cpu_data();
        const Dtype* label_data = bottom[1]->cpu_data();
        vector<int> bottom_shape = bottom[0]->shape();
        side_ = bottom_shape[2];
        const int pow_side = side_ * side_;
        const int size = pow_side * (5 + num_classes_);
        int index;
        
        Dtype* diff = diff_.mutable_cpu_data();
        Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), coord_loss(0.0), area_loss(0.0);
        Dtype avg_cat(0.0), noobj_score(0.0), obj_score(0.0), avg_best_iou(0.0), recall(0.0), recall75(0.0);
        int obj_count = 0;

        int locations = pow(side_, 2);
        int total_values = pow_side * num_boxes_ * (num_classes_ + 5);

       // LOGISTIC ACTIVATION
       for(int b = 0; b < bottom[0]->num(); ++b) {
           for(int n = 0; n < num_boxes_; ++n) {
               index = (b * total_values) + ((n*side_*side_ / pow_side) * pow_side * (5 + num_classes_)) + (4 * pow_side) + (n*side_*side_ % pow_side);
               for(int s = index; s < (index + (side_*side_*(num_classes_+1))); s++)
                   input_data[s] = logistic_activate(input_data[s]);
           }
       }
        //BBOX
        caffe_set(diff_.count(), Dtype(0.), diff);
        for(int batch = 0; batch < bottom[0]->num(); ++batch)
        {
            int truth_index =  batch * bottom[1]->count(1);
            std::vector<caffe::Box<Dtype> > true_box_list;
            for(int j = 0; j < locations; ++j){
               bool isobj = label_data[locations + truth_index + j];
               if(!isobj) continue;
               true_box_list.push_back(dtype_to_box(label_data + truth_index + locations * 3 + j * 4));
            }
            for(int j = 0; j < locations; ++j){
                for(int n = 0; n < num_boxes_; ++n){
                    int boxes_index = j + size * n + batch * bottom[0]->count(1);
                    Box<Dtype> pred_box = get_yolo_box(input_data + boxes_index, 
                        biases_, masks_[n], pow_side, j % side_, j / side_, net_w_, net_h_, side_);
                    float best_iou = 0;
                    int best_t = 0;
                    for(int index = 0; index < true_box_list.size(); ++index) {
                        float iou = Calc_iou(pred_box, true_box_list[index]);
                        if(iou > best_iou) {
                            best_iou = iou;
                            best_t = index;
                        }
                    }
                    int obj_index = boxes_index + 4 * pow_side;
                    noobj_score += input_data[obj_index];
                    noobj_loss += pow(input_data[obj_index], 2);
                    diff[obj_index] = (-1) * ((0 - input_data[obj_index]) * logistic_gradient(input_data[obj_index]));
                    if(best_iou > ignore_thresh_) {
                        diff[obj_index] = 0;
                    }
                    if (best_iou > truth_thresh_) {
                        diff[obj_index] = (-1) * ((1 - input_data[obj_index]) * logistic_gradient(input_data[obj_index]));
                        int w = true_box_list[best_t].x * side_;
                        int h = true_box_list[best_t].y * side_;
                        int label = static_cast<int>(label_data[locations * 2 + truth_index + h * side_ + w]);
                        int class_start_index = obj_index + pow_side;
                        if (diff[class_start_index]) {
                            diff[class_start_index + pow_side*label] = (-1) * (1 - input_data[class_start_index + pow_side*label]);
                        }
                        else {
                            for(int index = 0; index < num_classes_; ++index) {
                                int class_index = obj_index + pow_side * (index + 1);
                                Dtype target(index == label);
                                diff[class_index] = (-1) * (target - input_data[class_index]);
                                class_loss += pow((target - input_data[class_index]), 2);
                            }
                        }
                        delta_yolo_box(true_box_list[best_t], input_data, biases_, masks_[n],
                                side_, diff, boxes_index, 
                                    (2 - true_box_list[best_t].w *
                                            true_box_list[best_t].h), j % side_, j / side_, net_w_, net_h_, coord_loss, area_loss);
                    }
                }
            }
            for(int index = 0; index < true_box_list.size(); ++index){
                float best_iou = 0;
                int best_n = 0;
                Box<Dtype> shift_box = true_box_list[index];
                shift_box.x = shift_box.y = 0;
                int w = true_box_list[index].x * side_;
                int h = true_box_list[index].y * side_;
                for(int n = 0; n < total_num_boxes_; ++n) {
                    Box<Dtype> pred_box = {0};
                    pred_box.x = pred_box.y = 0.0;
                    pred_box.w = static_cast<Dtype>(biases_[2 * n]) / net_w_;
                    pred_box.h = static_cast<Dtype>(biases_[2 * n + 1]) / net_h_;
                    float iou = Calc_iou(pred_box, shift_box);
                    if(iou > best_iou) {
                        best_iou = iou;
                        best_n = n;
                    }
                }
                int mask_n = int_index(masks_, best_n, num_boxes_);
                if(mask_n >= 0) {
                  int box_index = (h * side_ + w) + size * mask_n + batch * bottom[0]->count(1);
                  float iou = delta_yolo_box(true_box_list[index], input_data, biases_,
                                best_n, side_, diff, box_index, 
                                    (2 - true_box_list[index].w *
                                            true_box_list[index].h), w, h, net_w_, net_h_, coord_loss, area_loss);
                  int obj_index = box_index + 4 * pow_side;
                  obj_score += input_data[obj_index];
                  obj_loss += pow((1 - input_data[obj_index]), 2);
                  diff[obj_index] = (-1) * (1 - input_data[obj_index]) * logistic_gradient(input_data[obj_index]);

                  int label = static_cast<int>(label_data[locations * 2 + truth_index + h * side_ + w]);
                  int class_start_index = obj_index + pow_side;
                  if (diff[class_start_index]) {
                      diff[class_start_index + pow_side*label] = (-1) * (1 - input_data[class_start_index + pow_side*label]);
                      avg_cat += input_data[class_start_index + pow_side*label];
                  }
                  else {
                      for(int index = 0; index < num_classes_; ++index) {
                          int class_index = obj_index + pow_side * (index + 1);
                          Dtype target(index == label);
                          diff[class_index] = (-1) * (target - input_data[class_index]);
                          class_loss += pow((target - input_data[class_index]), 2);
                          if(target) {
                              avg_cat += input_data[class_index];
                          }
                      }
                  }
                obj_count += 1;
                if(iou > .5) ++recall;
                if(iou > .75) ++recall75;
                avg_best_iou += iou;
                }
            }
        }
      
        int diff_size = bottom[0]->num() * pow_side * num_boxes_;
        class_loss /= obj_count;
        coord_loss /= obj_count;
        area_loss /= obj_count;
        obj_loss /= obj_count;
        noobj_loss /= (diff_size - obj_count);
        
        avg_best_iou /= obj_count;
        avg_cat /= obj_count;
        obj_score /= obj_count;
        noobj_score /= (diff_size - obj_count);
        recall /= obj_count;
        recall75 /= obj_count;


        LOG(INFO) << "class_loss: " << avg_cat << " obj_score: " << obj_score
                  << " noobj_score: " << noobj_score << " avg_best_iou " << avg_best_iou
                  << " Avg Recall(0.5): " << recall << " Avg Recall(0.75): " << recall75 << " count: " << obj_count;
        loss = class_loss + coord_loss + area_loss + obj_loss + noobj_loss;
        top[0]->mutable_cpu_data()[0]  = loss;
        if(seen < SEEN_NUMBER) {
            LOG(INFO) << "seen is: " << seen;
            seen += bottom[0]->num();
        }
    }

    template <typename Dtype>
    void YoloV3LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        if (propagate_down[1]) {
          LOG(FATAL) << this->type()
                     << " Layer cannot backpropagate to label inputs.";
        }
        if (propagate_down[0]) {
          const Dtype sign(1.);
          const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
          LOG(INFO) << "alpha: " << alpha;
          caffe_cpu_axpby(
              bottom[0]->count(),
              alpha,
              diff_.cpu_data(),
              Dtype(0),
              bottom[0]->mutable_cpu_diff());
        }
    }


#ifdef CPU_ONLY
    //STUB_GPU(YoloV3LossLayer);
#endif
   
    INSTANTIATE_CLASS(YoloV3LossLayer);
    REGISTER_LAYER_CLASS(YoloV3Loss);
}

