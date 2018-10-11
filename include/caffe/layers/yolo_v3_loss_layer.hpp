#ifndef YOLOV3LOSSLAYER_H
#define YOLOV3LOSSLAYER_H

#include <vector>
#include <google/protobuf/repeated_field.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
    template <typename Dtype>
    Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);
    template <typename Dtype>
    Dtype Calc_iou(const std::vector<Dtype>& box, const std::vector<Dtype>& truth);

    template<typename Dtype>
    class YoloV3LossLayer: public LossLayer<Dtype> {
    public:
        explicit YoloV3LossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param), diff_() {}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top);

        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "YoloV3Loss"; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        vector<int> biases_;
        vector<int> masks_;
        int side_;
        int num_classes_;
        int num_boxes_;
        int total_num_boxes_;
        float ignore_thresh_;
        float truth_thresh_;
        int seen;
        int net_w_;
        int net_h_;

        Blob<Dtype> diff_;
    };
}

#endif // YOLOV3LOSSLAYER_H
