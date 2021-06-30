#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/tensor2box_layer.hpp"

namespace caffe {

template <typename Dtype>
void Tensor2BoxLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    const Tensor2BoxParameter &tensor2box_param =
        this->layer_param_.tensor2box_param();
    num_classes_ = tensor2box_param.num_classes();
    img_dim_h_ = tensor2box_param.img_dim_h();
    img_dim_w_ = tensor2box_param.img_dim_w();
    anchors_x_.clear();
    std::copy(tensor2box_param.anchors_x().begin(),
              tensor2box_param.anchors_x().end(),
              std::back_inserter(anchors_x_));
    anchors_y_.clear();
    std::copy(tensor2box_param.anchors_y().begin(),
              tensor2box_param.anchors_y().end(),
              std::back_inserter(anchors_y_));
    CHECK_EQ(anchors_x_.size(), anchors_y_.size())
        << "anchor_x and anchor_y should have the same length.";
}

template <typename Dtype>
void Tensor2BoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // python example has input = (1, 27, 20, 30)
    // and output = (1, 1800, 9)
    const int count = bottom[0]->count();
    // output shape = (num_samples, -1, 4+1+num_classes)
    //                (1,-1,9) = (1, 1800, 9)
    // while input shape = (1,27,20,30)
    vector<int> new_shape(3, 0); // (0, 0, 0)
    new_shape[0] = bottom[0]->shape(0);
    new_shape[2] = 4 + 1 + num_classes_;
    new_shape[1] = count / new_shape[0] / new_shape[2];
    top[0]->Reshape(new_shape);
    CHECK_EQ(count, top[0]->count());
}

//void comput_grid_offsets(int grid_size_h, int grid_size_w, int img_dim_h, int )
/*template <typename Dtype>
inline Dtype exp(Dtype x) {
	// std::exp(-x) for -x less than -87 will cause underflow 32bit float range
	//if (x < -86) return Dtype(0.0);
	return std::exp(x);
}*/

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
	return 0.5 * std::tanh(0.5 * x) + 0.5;
	// if use std::exp should clamp exp(-87 or lower), it will underflow 32bit float
    //return 1.0 / (1.0 + exp(-x));
}

template <typename Dtype>
void Tensor2BoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // this layer is translating tensor2box python function from
    // cnn_models/pytorch/Nikon/4thBenchmark/ScenarioForAI-CVSubsystem/subModules/detection.py#L97-176
    const Dtype *bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
    // setup variables
    const int num_anchors = anchors_x_.size();
    const int num_samples = bottom[0]->shape(0);
    const int H = bottom[0]->shape(2), W = bottom[0]->shape(3);
    const pair<int, int> grid_size(H, W);
    // 1st part:
    //   data.reshape(num_samples, num_anchors, num_classes + 5, grid_size[0], grid_size[1]).transpose(0, 1, 3, 4, 2)
    //   (1, 27, 20, 30) -> (1, 3, 20, 30, 9)
    // we can do the transpose operation during moving bottom_data to top_data
    const int _div = bottom[0]->shape(1) / num_anchors;
    int old_idx = 0;
    for (int s = 0; s < num_samples; ++s) {
        for (int a = 0; a < num_anchors; ++a) {
            for (int d = 0; d < _div; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        // move old_idx==(s,a*num_anchors+d,h,w) to new_idx==(s,a,h,w,d)
                        int new_idx = (((s*num_anchors+a)*H+h)*W+w)*_div+d;
                        top_data[new_idx] = bottom_data[old_idx++];
                    }
                }
            }
        }
    }
    // 2nd part: compute_grid_offsets
    const int stride_w = img_dim_w_ / W; // stride_w = img_dim[1] / grid_size[1]
    const int stride_h = img_dim_h_ / H; // stride_h = img_dim[0] / grid_size[0]
    vector<Dtype>anchor_h(num_anchors, 0.0); // to store scaled anchor_h and anchor_w
    vector<Dtype>anchor_w(num_anchors, 0.0); // to store scaled anchor_h and anchor_w
    for (int i = 0; i < num_anchors; ++i) {
        anchor_w[i] = Dtype(anchors_x_[i]) / stride_w;
        anchor_h[i] = Dtype(anchors_y_[i]) / stride_h;
    }
    // 3rd part: pred_boxes
    for (int s = 0; s < num_samples; ++s) {
        for (int a = 0; a < num_anchors; ++a) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    // 0,1,4,5~ sigmoid
                    // 0,1      +grid_xy
                    // 2,3      exp() * anchor
                    // 0~4      *stride
                    top_data[0] = (sigmoid(top_data[0]) + w) * stride_w;
                    top_data[1] = (sigmoid(top_data[1]) + h) * stride_h;
                    top_data[2] = (std::exp(top_data[2]) * anchor_w[a]) * stride_w;
                    top_data[3] = (std::exp(top_data[3]) * anchor_h[a]) * stride_h;
                    // pred_conf = sigmoid(prediction[..., 4])  # Conf
                    // pred_cls = sigmoid(prediction[..., 5:])  # Cls pred.
                    for (int i = 4; i < _div; ++i) {
                        top_data[i] = sigmoid(top_data[i]);
                    }
                    top_data += _div;
                }
            }
        }
    }
    ;//end of forward_cpu
}

INSTANTIATE_CLASS(Tensor2BoxLayer);
REGISTER_LAYER_CLASS(Tensor2Box);

}  // namespace caffe
