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
  this->data_format = this->layer_param_.resize_nearest_neighbor_param().data_format();
  this->output_height = this->layer_param_.resize_nearest_neighbor_param().output_height();
  this->output_width = this->layer_param_.resize_nearest_neighbor_param().output_width();
  this->half_pixel_centers = this->layer_param_.resize_nearest_neighbor_param().half_pixel_centers();
  this->half_pixel_onnx = this->layer_param_.resize_nearest_neighbor_param().half_pixel_onnx();
  CHECK_LE((this->align_corners + this->half_pixel_centers + this->half_pixel_onnx), 1) <<
      "Maximum one Flag in align_corners, half_pixel_center or half_pixel_onnx could be True.";
}

template <typename Dtype>
void ResizeNearestNeighborLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  const int batch_size = bottom_shape[0];
  if(this->data_format == "NHWC"){
    const int channels = bottom_shape[3];
    if(!this->layer_param_.resize_nearest_neighbor_param().has_output_height())
     {
       float scale_height = this->layer_param_.resize_nearest_neighbor_param().scale_height();
       this->output_height = floor(bottom_shape[1] * scale_height);
     }
     if(!this->layer_param_.resize_nearest_neighbor_param().has_output_width())
     {
       float scale_width = this->layer_param_.resize_nearest_neighbor_param().scale_width();
       this->output_width = floor(bottom_shape[2] * scale_width);
     }
    top[0]->Reshape(batch_size, this->output_height, this->output_width, channels);
  } else {
    const int channels = bottom_shape[1];
    if(!this->layer_param_.resize_nearest_neighbor_param().has_output_height())
    {
      float scale_height = this->layer_param_.resize_nearest_neighbor_param().scale_height();
      this->output_height = floor(bottom_shape[2] * scale_height);
    }
    if(!this->layer_param_.resize_nearest_neighbor_param().has_output_width())
    {
      float scale_width = this->layer_param_.resize_nearest_neighbor_param().scale_width();
      this->output_width = floor(bottom_shape[3] * scale_width);
    }
    top[0]->Reshape(batch_size, channels, this->output_height, this->output_width);
  }
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
  if(this->data_format == "NHWC"){
    const int input_height = bottom_shape[1];
    const int input_width = bottom_shape[2];
    const int channels = bottom_shape[3];

    vector<int> top_shape = top[0]->shape();
    const int output_height = top_shape[1];
    const int output_width = top_shape[2];

    const bool align_corners = this->align_corners;
    const bool half_pixel_centers = this->half_pixel_centers;
    const bool half_pixel_onnx = this->half_pixel_onnx;

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    //const Dtype* resize_shape = bottom[1]->cpu_data();
    //const int out_height = resize_shape[0];
    //const int out_width = resize_shape[1];
    //CHECK_EQ(out_height, output_height) << "output_height must equal to the bottom 1's height value "<<out_height<<".\n";
    //CHECK_EQ(out_width, output_width) << "output_width must equal to the bottom 1's width value "<<out_width<<".\n";

    const float height_scale =
      CalculateResizeScale(output_height, input_height, align_corners);
    const float width_scale =
      CalculateResizeScale(output_width, input_width, align_corners);
    //LOG(INFO)<<height_scale<<" "<<width_scale<<"\n";

    //it implies NHWC data format
    for (int b = 0; b < batch_size; b++) {
      for (int h = 0; h < output_height; h++) {
        int in_h;
        if (half_pixel_onnx)
        {
          in_h = std::max(std::min(static_cast<int>(
              floorf((static_cast<float>(h) + 0.5f) * height_scale - 0.5)),
              input_height - 1),
              0);
        }
        else if (half_pixel_centers)
        {
          in_h = std::max(std::min(static_cast<int>(
              floorf((static_cast<float>(h) + 0.5f) * height_scale)),
              input_height - 1),
              0);
        }
        else
        {
          in_h = std::min((align_corners)
                       ? static_cast<int>(roundf(h * height_scale))
                           : static_cast<int>(floorf(h * height_scale)),
                             input_height - 1);
        }
        for (int w = 0; w < output_width; w++) {
          int in_w;
          if (half_pixel_onnx)
          {
            in_w = std::max(std::min(static_cast<int>(
                floorf((static_cast<float>(w) + 0.5f) * width_scale - 0.5)),
                input_width - 1),
                0);
          }
          else if (half_pixel_centers)
          {
            in_w = std::max(std::min(static_cast<int>(
                floorf((static_cast<float>(w) + 0.5f) * width_scale)),
                input_width - 1),
                0);
          }
          else
          {
            in_w = std::min((align_corners)
                ? static_cast<int>(roundf(w * width_scale))
                    : static_cast<int>(floorf(w * width_scale)),
                      input_width - 1);
          }
          for(int c = 0; c < channels; c++) {
            const int input_index = ((b*input_height + in_h)*input_width + in_w)*channels + c;
            const int output_index = ((b*output_height + h)*output_width + w)*channels + c;
            top_data[output_index] = bottom_data[input_index];
            //LOG(INFO)<<output_index<<" "<<input_index<<"; "<<top_data[output_index]<<" "<<bottom_data[input_index]<<"; "<<b<<"; "<<h<<" "<<in_h<<"; "<<w<<" "<<in_w<<"; "<<c<<" "<<"\n";
          }
        }
      }
    }
  } else {
    const int channels = bottom_shape[1];
    const int input_height = bottom_shape[2];
    const int input_width = bottom_shape[3];

    vector<int> top_shape = top[0]->shape();
    const int output_height = top_shape[2];
    const int output_width = top_shape[3];

    const bool align_corners = this->align_corners;
    const bool half_pixel_centers = this->half_pixel_centers;
    const bool half_pixel_onnx = this->half_pixel_onnx;

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    const float height_scale =
      CalculateResizeScale(output_height, input_height, align_corners);
    const float width_scale =
      CalculateResizeScale(output_width, input_width, align_corners);

    //LOG(INFO)<<height_scale<<" "<<width_scale<<std::endl;

    //it implies NCHW data format
    for (int b = 0; b < batch_size; b++) {
      for(int c = 0; c < channels; c++) {
        for (int h = 0; h < output_height; h++) {
          int in_h;
          if (half_pixel_onnx)
          {
            in_h = std::max(std::min(static_cast<int>(
                floorf((static_cast<float>(h) + 0.5f) * height_scale - 0.5)),
                input_height - 1),
                0);
          }
          else if (half_pixel_centers)
          {
            in_h = std::max(std::min(static_cast<int>(
                floorf((static_cast<float>(h) + 0.5f) * height_scale)),
                input_height - 1),
                0);
          }
          else
          {
            in_h = std::min((align_corners)
                ? static_cast<int>(roundf(h * height_scale))
                    : static_cast<int>(floorf(h * height_scale)),
                      input_height - 1);
          }
          for (int w = 0; w < output_width; w++) {
            int in_w;
            if (half_pixel_onnx)
            {
              in_w = std::max(std::min(static_cast<int>(
                  floorf((static_cast<float>(w) + 0.5f) * width_scale - 0.5)),
                  input_width - 1),
                  0);
            }
            else if (half_pixel_centers)
            {
              in_w = std::max(std::min(static_cast<int>(
                  floorf((static_cast<float>(w) + 0.5f) * width_scale)),
                  input_width - 1),
                  0);
            }
            else
            {
              in_w = std::min((align_corners)
                  ? static_cast<int>(roundf(w * width_scale))
                      : static_cast<int>(floorf(w * width_scale)),
                        input_width - 1);
            }
            //LOG(INFO)<<w<<" "<<in_w<<std::endl;
            const int input_index = ((b*channels + c)*input_height + in_h)*input_width + in_w;
            const int output_index = ((b*channels + c)*output_height + h)*output_width + w;
            top_data[output_index] = bottom_data[input_index];
          }
        }
      }
    }
  }

}

INSTANTIATE_CLASS(ResizeNearestNeighborLayer);
REGISTER_LAYER_CLASS(ResizeNearestNeighbor);

}  // namespace caffe
