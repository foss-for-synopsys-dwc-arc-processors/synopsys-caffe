#include <vector>

#include "caffe/layers/spacetodepth_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpaceToDepthLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  this->block_size = this->layer_param_.space_to_depth_param().block_size();
  this->data_format = this->layer_param_.space_to_depth_param().data_format();
  vector<int> bottom_shape = bottom[0]->shape();
  this->output_top_shape.push_back(bottom_shape[0]);

  if(this->data_format == "NHWC"){
    this->output_top_shape.push_back(bottom_shape[1] / this->block_size);
    this->output_top_shape.push_back(bottom_shape[2] / this->block_size);
    this->output_top_shape.push_back(bottom_shape[3] * (this->block_size*this->block_size));
  } else if(this->data_format == "NCHW"){
    this->output_top_shape.push_back(bottom_shape[1] * (this->block_size*this->block_size));
    this->output_top_shape.push_back(bottom_shape[2] / this->block_size);
    this->output_top_shape.push_back(bottom_shape[3] / this->block_size);
  }
}

template <typename Dtype>
void SpaceToDepthLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(this->output_top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count())
    << "output count must match input count";
}

template <typename Dtype>
void SpaceToDepthLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  if (this->data_format == "NHWC"){
    const int batch_size = this->output_top_shape[0];
    const int output_height = this->output_top_shape[1];
    const int output_width = this->output_top_shape[2];
    const int output_depth = this->output_top_shape[3];

    vector<int> bottom_shape = bottom[0]->shape();
    const int input_height = bottom_shape[1];
    const int input_width = bottom_shape[2];
    const int input_depth = bottom_shape[3];

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < input_height; ++h) {
        const int out_h = h / this->block_size;
        const int offset_h = (h % this->block_size);
        for (int w = 0; w < input_width; ++w) {
          const int out_w = w / this->block_size;
          const int offset_w = (w % this->block_size);
          const int offset_d =
            (offset_h * this->block_size + offset_w) * input_depth;
          for (int d = 0; d < input_depth; ++d) {
            const int out_d = d + offset_d;
            const int in_index = ((b*input_height + h)*input_width + w)*input_depth + d;
            const int out_index = ((b*output_height + out_h)*output_width + out_w)*output_depth + out_d;
            top_data[out_index] = bottom_data[in_index];
          }
        }
      }
    }
  } else {
    const int batch_size = this->output_top_shape[0];
    const int output_depth = this->output_top_shape[1];
    const int output_height = this->output_top_shape[2];
    const int output_width = this->output_top_shape[3];

    vector<int> bottom_shape = bottom[0]->shape();
    const int input_depth = bottom_shape[1];
    const int input_height = bottom_shape[2];
    const int input_width = bottom_shape[3];

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < input_height; ++h) {
        const int out_h = h / this->block_size;
        const int offset_h = (h % this->block_size);
        for (int w = 0; w < input_width; ++w) {
          const int out_w = w / this->block_size;
          const int offset_w = (w % this->block_size);
          const int offset_d =
            (offset_h * this->block_size + offset_w) * input_depth;
          for (int d = 0; d < input_depth; ++d) {
            const int out_d = d + offset_d;
            const int in_index = ((b*input_depth + d)*input_height + h)*input_width + w;
            const int out_index = ((b*output_depth + out_d)*output_height + out_h)*output_width + out_w;
            top_data[out_index] = bottom_data[in_index];
          }
        }
      }
    }
  }
}


INSTANTIATE_CLASS(SpaceToDepthLayer);
REGISTER_LAYER_CLASS(SpaceToDepth);

}  // namespace caffe
