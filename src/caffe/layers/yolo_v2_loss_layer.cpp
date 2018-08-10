#include "caffe/layers/yolo_v2_loss_layer.hpp"
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
    Box<Dtype> get_region_box(const Dtype *x, vector<float>& biases, int n, int SIZE, int i, int j, int side){
        Box<Dtype> box;
        box.x = (i + logistic_activate(x[0 * SIZE])) / side;
        box.y = (j + logistic_activate(x[1 * SIZE])) / side;
        box.w = exp(x[2 * SIZE]) * biases[2 * n] / side;
        box.h = exp(x[3 * SIZE]) * biases[2 * n + 1] / side;
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
    Dtype abs(Dtype x) {
        if(x < 0)
            return -x;
        return x;
    }

    template <typename Dtype>
    Dtype Calc_rmse(const Box<Dtype>& truth, const Box<Dtype>& box, Dtype &coord_loss, Dtype &area_loss, float scale) {
        float coord_ = scale * (abs(box.x-truth.x) + abs(box.y-truth.y));
        float area_  = scale * (abs(box.w-truth.w) + abs(box.h-truth.h));
        coord_loss += coord_;
        area_loss  += area_;
        return (coord_ + area_);
    }

   template <typename Dtype>
    float delta_region_box(Box<Dtype>& truth_box, Dtype* input, 
            std::vector<float>& biases, int n, int side, Dtype* diff, 
            int index, float scale, int i, int j, Dtype &coord_loss, Dtype &area_loss) {
        
        Box<Dtype> pred_box = get_region_box(input + index, biases, n, side*side, i, j, side);
        float iou = Calc_iou(pred_box, truth_box);

        float tx = (truth_box.x*side - i);
        float ty = (truth_box.y*side - j);
        float tw = log((truth_box.w*side) / biases[2*n]);
        float th = log((truth_box.h*side) / biases[2*n + 1]);
        
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
    void YoloV2LossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);

        YoloV2LossParameter param = this->layer_param().yolo_v2_loss_param();
        const google::protobuf::RepeatedField<float> &tmpBiases_ = param.anchors();
        for(google::protobuf::RepeatedField<float>::const_iterator 
            iterator = tmpBiases_.begin(); iterator != tmpBiases_.end(); ++iterator)
        {
            biases_.push_back(*iterator);
            //LOG(INFO) << *iterator;
        }

        side_           = param.side();
        CHECK_GE(side_, 0) << "side size must bigger then 0";
        num_classes_    = param.num_classes();
        CHECK_GE(num_classes_, 0) << "class number must bigger then 0";
        num_boxes_      = param.num_object();
        CHECK_GE(num_boxes_, 0) << "box number must bigger then 0";
        CHECK_EQ(biases_.size() , num_boxes_ * 2) << "biases size and num_boxes doesn't match";
        seen            = 0;
        box_scale_      = param.box_scale();
        class_scale_    = param.class_scale();
        object_scale_   = param.object_scale();
        noobject_scale_ = param.noobject_scale();
        rescore_        = param.rescore();
        constraint_     = param.constraint();
        thresh_         = param.thresh();

        int input_count = bottom[0]->count(1);
        int label_count = bottom[1]->count(1);
        int tmp_input_count = side_ * side_ * (num_classes_ + (1 + 4)) * num_boxes_;
        int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);
        CHECK_EQ(input_count, tmp_input_count);
        CHECK_EQ(label_count, tmp_label_count);
    }

    template <typename Dtype>
    void YoloV2LossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::Reshape(bottom, top);
        diff_.ReshapeLike(*bottom[0]);
    }

    int entry_index(int pow_side, int coords_class, int total_values, int batch, int location, int entry)
    {
        int n =   location / (pow_side);
        int loc = location % (pow_side);
        return (batch * total_values) + (n * pow_side * coords_class) + (entry * pow_side) + loc;
    }

    template <typename Dtype>
    void softmax_yolo(Dtype *input, int n, float temp, int stride, Dtype *output)
    {
        int i;
        float sum = 0;
        float largest = -FLT_MAX;
        for(i = 0; i < n; ++i){
            if(input[i*stride] > largest) largest = input[i*stride];
        }
        for(i = 0; i < n; ++i){
            float e = exp(input[i*stride]/temp - largest/temp);
            sum += e;
            output[i*stride] = e;
        }
        for(i = 0; i < n; ++i){
            output[i*stride] /= sum;
        }
    }

    template <typename Dtype>
    void softmax_cpu(Dtype *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, Dtype *output)
    {
        int g, b;
        for(b = 0; b < batch; ++b){
            for(g = 0; g < groups; ++g){
                softmax_yolo(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
            }
        }
    }

    template <typename Dtype>
    void YoloV2LossLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        Dtype* input_data = bottom[0]->mutable_cpu_data();
        const Dtype* label_data = bottom[1]->cpu_data();
        const int pow_side = side_ * side_;
        const int size = pow_side * (5 + num_classes_);
        int index;
        
        Dtype* diff = diff_.mutable_cpu_data();
        Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), coord_loss(0.0), area_loss(0.0);
        Dtype avg_cat(0.0), noobj_score(0.0), obj_score(0.0), avg_best_iou(0.0), recall(0.0);
        int obj_count = 0;

        int locations = pow(side_, 2);
        int total_values = pow_side * num_boxes_ * (num_classes_ + 5);

       // LOGISTIC ACTIVATION
       for(int b = 0; b < bottom[0]->num(); ++b) {
           for(int n = 0; n < num_boxes_; ++n) {
               index = entry_index(pow_side, (5 + num_classes_), total_values, b, n*side_*side_, 4);
               for(int s = index; s < (index + (side_*side_)); s++)
                 input_data[s] = logistic_activate(input_data[s]);
           }
       }

        // SOFTMAX
        index = entry_index(pow_side, (5 + num_classes_), total_values, 0, 0, 5);// coords + !background
        softmax_cpu(input_data + index, num_classes_, bottom[0]->num()*num_boxes_, total_values/num_boxes_, pow_side, 1, pow_side, 1, input_data + index);

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
                    Box<Dtype> pred_box = get_region_box(input_data + boxes_index, 
                        biases_, n, pow_side, j % side_, j / side_, side_);
                    float best_iou = 0;
                    for(int index = 0; index < true_box_list.size(); ++index) {
                    
                        float iou = Calc_iou(pred_box, true_box_list[index]);
                        if(iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = boxes_index + 4 * pow_side;
                    noobj_score += input_data[obj_index];
                    
                    if(best_iou > thresh_) {
                        diff[obj_index] = 0;
                    }
                    else {
                        noobj_loss += noobject_scale_ * pow(input_data[obj_index], 2);
                        diff[obj_index] = (-1) * noobject_scale_ * ((0 - input_data[obj_index]) * logistic_gradient(input_data[obj_index]));
                    }
                    if(seen < SEEN_NUMBER) {
                        Box<Dtype> truth;
                        truth.x = ((j % side_) + .5) / side_;
                        truth.y = ((j / side_) + .5) / side_;
                        truth.w = biases_[2 * n + 0] / side_;
                        truth.h = biases_[2 * n + 1] / side_;
                        delta_region_box(truth, input_data, biases_, n, side_, diff,
                            boxes_index, .01, j % side_, j / side_, coord_loss, area_loss);
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

                for(int n = 0; n < num_boxes_; ++n) {
                    int box_index = (h * side_ + w) + size * n + batch * bottom[0]->count(1);

                    Box<Dtype> pred_box = get_region_box(input_data + box_index, 
                        biases_, n, pow_side, w, h, side_);
                    pred_box.x = pred_box.y = 0;
                    bool bias_match = true; //TODO
                    if(bias_match) {
                        pred_box.w = biases_[2 * n + 0] / side_;
                        pred_box.h = biases_[2 * n + 1] / side_;
                    }
                    float iou = Calc_iou(pred_box, shift_box);
                    if(iou > best_iou) {
                        best_iou = iou;
                        best_n = n;
                    }
                }

                int box_index = (h * side_ + w) + size * best_n + batch * bottom[0]->count(1);
                float iou = delta_region_box(true_box_list[index], input_data, biases_,
                                best_n, side_, diff, box_index, 
                                    box_scale_ * (2 - true_box_list[index].w *
                                            true_box_list[index].h), w, h, coord_loss, area_loss);

                if(iou > .5) ++recall;
                avg_best_iou += iou;

                int obj_index = box_index + 4 * pow_side;
                obj_score += input_data[obj_index];
                
                
                
                if (rescore_) {
                  obj_loss += object_scale_ * pow(iou - input_data[obj_index], 2);
                  diff[obj_index] = (-1) * object_scale_ * (iou - input_data[obj_index]) * logistic_gradient(input_data[obj_index]);
                }
                else {
                  obj_loss += object_scale_ * pow(1 - input_data[obj_index], 2);
                  diff[obj_index] = (-1) * object_scale_ * (1 - input_data[obj_index]) * logistic_gradient(input_data[obj_index]);
                }

                int label = static_cast<int>(label_data[locations * 2 + truth_index + h * side_ + w]);
                for(int index = 0; index < num_classes_; ++index) {
                    int class_index = obj_index + pow_side * (index + 1);
                    Dtype target(index == label);
                    diff[class_index] = (-1) * class_scale_ * (target - input_data[class_index]);
                    class_loss += class_scale_ * pow((target - input_data[class_index]), 2);
                    if(target) {
                        avg_cat += input_data[class_index];
                    }
                }
            }
            obj_count += true_box_list.size();
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


        LOG(INFO) << "class_loss: " << avg_cat << " obj_score: " << obj_score
                  << " noobj_score: " << noobj_score << " avg_best_iou " << avg_best_iou
                  << " Avg Recall: " << recall << " count: " << obj_count;
        loss = class_loss + coord_loss + area_loss + obj_loss + noobj_loss;
        top[0]->mutable_cpu_data()[0]  = loss;

        if(seen < SEEN_NUMBER) {
            LOG(INFO) << "seen is: " << seen;
            seen += bottom[0]->num();
        }
    }

    template <typename Dtype>
    void YoloV2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    STUB_GPU(YoloV2LossLayer);
#endif
   
    INSTANTIATE_CLASS(YoloV2LossLayer);
    REGISTER_LAYER_CLASS(YoloV2Loss);
}

