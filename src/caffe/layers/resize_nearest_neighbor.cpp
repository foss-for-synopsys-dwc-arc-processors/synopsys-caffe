#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/layers/resize_nearest_neighbor_layer.hpp"

namespace caffe {

template <typename Dtype>
void ResizeNearestNeighborLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  this->align_corners = this->layer_param_.resize_nearest_neighbor_param().align_corners();
}

template <typename Dtype>
void ResizeNearestNeighborLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(top[1]->shape());
}

// CalculateResizeScale determines the float scaling factor.
inline float CalculateResizeScale(int in_size, int out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size);
}
template <typename Dtype>
void ResizeNearestNeighborLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  vector<int> bottom_shape = bottom[0]->shape();
  const int batch_size = bottom_shape[0];
  const int input_height = bottom_shape[1];
  const int input_width = bottom_shape[2];
  const int channels = bottom_shape[3];

  const int output_height = bottom[1]->cpu_data()[0];
  const int output_width = bottom[1]->cpu_data()[1];
  const bool align_corners = this->align_corners;
  const float height_scale = CalculateResizeScale(output_height, input_height, align_corners);
  const float width_scale = CalculateResizeScale(output_width, input_width, align_corners);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  //it implies NHWC data format
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < output_height; ++h) {
      const int out_h = std::min((align_corners)
                                ? static_cast<int>(roundf(h * height_scale))
                                : static_cast<int>(floorf(h * height_scale)),
                                input_height - 1);
      for (int w = 0; w < output_width; ++w) {
        const int out_w = std::min((align_corners)
                                  ? static_cast<int>(roundf(h * width_scale))
                                  : static_cast<int>(floorf(h * width_scale)),
                                  input_width - 1);
        const int input_index = b*input_height*input_width*channels + out_h*input_width*channels + out_w*channels;
        const int output_index = b*output_height*output_width*channels + h*output_width*channels + w*channels;
        std::copy_n(&bottom_data[input_index], channels, &top_data[output_index]);
        }
      }
    }
  }

INSTANTIATE_CLASS(ResizeNearestNeighborLayer);
REGISTER_LAYER_CLASS(ResizeNearestNeighbor);

}  // namespace caffe
