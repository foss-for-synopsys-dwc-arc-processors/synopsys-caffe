#include <vector>

#include "caffe/layers/depth_to_space_layer.hpp"

namespace caffe {

template <typename Dtype>
void DepthToSpaceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
                                                 "allow in-place computation.";
  this->block_size = this->layer_param_.depth_to_space_param().block_size();
  this->data_format = this->layer_param_.depth_to_space_param().data_format();
  vector<int> bottom_shape = bottom[0]->shape();
  this->output_top_shape.push_back(bottom_shape[0]);
  if (this->data_format == "NHWC") {
    this->output_top_shape.push_back(bottom_shape[1] * this->block_size);
    this->output_top_shape.push_back(bottom_shape[2] * this->block_size);
    this->output_top_shape.push_back(bottom_shape[3] /
                                     (this->block_size * this->block_size));
  } else if (this->data_format == "NCHW" || this->data_format == "CRD" ||
             this->data_format == "DCR") {
    this->output_top_shape.push_back(bottom_shape[1] /
                                     (this->block_size * this->block_size));
    this->output_top_shape.push_back(bottom_shape[2] * this->block_size);
    this->output_top_shape.push_back(bottom_shape[3] * this->block_size);
  }
}

template <typename Dtype>
void DepthToSpaceLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  top[0]->Reshape(this->output_top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count())
      << "output count must match input count";
}

template <typename Dtype>
void DepthToSpaceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  if (this->data_format == "NHWC") {
    const int batch_size = this->output_top_shape[0];
    const int output_height = this->output_top_shape[1];
    const int output_width = this->output_top_shape[2];
    const int output_depth = this->output_top_shape[3];

    vector<int> bottom_shape = bottom[0]->shape();
    const int input_height = bottom_shape[1];
    const int input_width = bottom_shape[2];
    const int input_depth = bottom_shape[3];

    const Dtype *bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < output_height; ++h) {
        const int in_h = h / this->block_size;
        const int offset_h = (h % this->block_size);
        for (int w = 0; w < output_width; ++w) {
          const int in_w = w / this->block_size;
          const int offset_w = (w % this->block_size);
          const int offset_d =
              (offset_h * this->block_size + offset_w) * output_depth;
          for (int d = 0; d < output_depth; ++d) {
            const int in_d = d + offset_d;
            const int out_index =
                ((b * output_height + h) * output_width + w) * output_depth + d;
            const int in_index =
                ((b * input_height + in_h) * input_width + in_w) * input_depth +
                in_d;
            top_data[out_index] = bottom_data[in_index];
          }
        }
      }
    }
  } else if (this->data_format == "NCHW" || this->data_format == "DCR") {
    const int batch_size = this->output_top_shape[0];
    const int output_depth = this->output_top_shape[1];
    const int output_height = this->output_top_shape[2];
    const int output_width = this->output_top_shape[3];

    vector<int> bottom_shape = bottom[0]->shape();
    const int input_depth = bottom_shape[1];
    const int input_height = bottom_shape[2];
    const int input_width = bottom_shape[3];

    const Dtype *bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < output_height; ++h) {
        const int in_h = h / this->block_size;
        const int offset_h = (h % this->block_size);
        for (int w = 0; w < output_width; ++w) {
          const int in_w = w / this->block_size;
          const int offset_w = (w % this->block_size);
          const int offset_d =
              (offset_h * this->block_size + offset_w) * output_depth;
          for (int d = 0; d < output_depth; ++d) {
            const int in_d = d + offset_d;
            const int out_index =
                ((b * output_depth + d) * output_height + h) * output_width + w;
            const int in_index =
                ((b * input_depth + in_d) * input_height + in_h) * input_width +
                in_w;
            top_data[out_index] = bottom_data[in_index];
          }
        }
      }
    }
  } else if (this->data_format == "CRD") {
    const int batch_size = this->output_top_shape[0];
    const int output_depth = this->output_top_shape[1];
    const int output_height = this->output_top_shape[2];
    const int output_width = this->output_top_shape[3];

    vector<int> bottom_shape = bottom[0]->shape();
    const int input_depth = bottom_shape[1];
    const int input_height = bottom_shape[2];
    const int input_width = bottom_shape[3];

    const Dtype *bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < output_height; ++h) {
        const int in_h = h / this->block_size;
        const int offset_h = (h % this->block_size);
        for (int w = 0; w < output_width; ++w) {
          const int in_w = w / this->block_size;
          const int offset_w = (w % this->block_size);
          for (int d = 0; d < output_depth; ++d) {
            const int in_d =
                (d * this->block_size + offset_h) * this->block_size + offset_w;
            const int out_index =
                ((b * output_depth + d) * output_height + h) * output_width + w;
            const int in_index =
                ((b * input_depth + in_d) * input_height + in_h) * input_width +
                in_w;
            top_data[out_index] = bottom_data[in_index];
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(DepthToSpaceLayer);
REGISTER_LAYER_CLASS(DepthToSpace);

} // namespace caffe
