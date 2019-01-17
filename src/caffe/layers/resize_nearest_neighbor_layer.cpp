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
  this->output_height = this->layer_param_.resize_nearest_neighbor_param().output_height();
  this->output_width = this->layer_param_.resize_nearest_neighbor_param().output_width();
}

template <typename Dtype>
void ResizeNearestNeighborLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  const int batch_size = bottom_shape[0];
  const int channels = bottom_shape[3];
  // Assume the data in NHWC format
  top[0]->Reshape(batch_size, this->output_height, this->output_width, channels);
}

// CalculateResizeScale computes the float scaling factor for height and width.
inline float CalculateResizeScale(int out_size, int in_size, bool align_corners) {
  return ( (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size) );
}

template <typename Dtype>
void ResizeNearestNeighborLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  vector<int> bottom_shape = bottom[0]->shape();
  const int batch_size = bottom_shape[0];
  const int input_height = bottom_shape[1];
  const int input_width = bottom_shape[2];
  const int channels = bottom_shape[3];

  vector<int> top_shape = top[0]->shape();
  const int output_height = top_shape[1];
  const int output_width = top_shape[2];

  const bool align_corners = this->align_corners;

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* resize_shape = bottom[1]->cpu_data();
  const int out_height = resize_shape[0];
  const int out_width = resize_shape[1];
  CHECK_EQ(out_height, output_height) << "output_height must equal to the bottom 1's height value "<<out_height<<".\n";
  CHECK_EQ(out_width, output_width) << "output_width must equal to the bottom 1's width value "<<out_width<<".\n";

  const float height_scale =
      CalculateResizeScale(output_height, input_height, align_corners);
  const float width_scale =
      CalculateResizeScale(output_width, input_width, align_corners);
  //LOG(INFO)<<height_scale<<" "<<width_scale<<"\n";

  //it implies NHWC data format
  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < output_height; h++) {
      const int in_h = std::min( (align_corners)
                                ? static_cast<int>(roundf(h * height_scale))
                                : static_cast<int>(floorf(h * height_scale)),
                        input_height - 1);
      for (int w = 0; w < output_width; w++) {
        const int in_w = std::min( (align_corners)
                                  ? static_cast<int>(roundf(w * width_scale))
                                  : static_cast<int>(floorf(w * width_scale)),
                          input_width - 1);
        for(int c = 0; c < channels; c++) {
          const int input_index = b*input_height*input_width*channels + in_h*input_width*channels + in_w*channels + c;
          const int output_index = b*output_height*output_width*channels + h*output_width*channels + w*channels + c;
          top_data[output_index] = bottom_data[input_index];
          //LOG(INFO)<<output_index<<" "<<input_index<<"; "<<top_data[output_index]<<" "<<bottom_data[input_index]<<"; "<<b<<"; "<<h<<" "<<in_h<<"; "<<w<<" "<<in_w<<"; "<<c<<" "<<"\n";
        }
      }
    }
  }
}

INSTANTIATE_CLASS(ResizeNearestNeighborLayer);
REGISTER_LAYER_CLASS(ResizeNearestNeighbor);

}  // namespace caffe
