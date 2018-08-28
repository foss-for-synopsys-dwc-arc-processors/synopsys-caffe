#include <vector>

#include "caffe/layers/im2col_layer.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  const int input_num_dims = bottom[0]->shape().size();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_dim = channel_axis_ + 1;
  num_spatial_axes_ = input_num_dims - first_spatial_dim;
  CHECK_GE(num_spatial_axes_, 1);
  vector<int> dim_blob_shape(1, num_spatial_axes_);
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }

  // Setup pad dimensions (pad_).
  pad_.Reshape(dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }

  //<--CUSTOMIZATION
  if (conv_param.has_pad_type()){
    pad_type_ = conv_param.pad_type();
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_type should be specified; not both.";
    CHECK_EQ(0, conv_param.pad_h())
        << "Either pad_h/w or pad_type should be specified; not both.";
    CHECK_EQ(0, conv_param.pad_w())
        << "Either pad_h/w or pad_type should be specified; not both.";
    LOG(INFO) << "Note parameter pad_type is DEPRECATED. Please use pad_l/r/t/b instead.";
  } else{
	pad_type_ = 0;
  }

  if (conv_param.has_pad_l()){
    pad_l_ = conv_param.pad_l();
  }
  else{
    pad_l_ = 0;
  }
  if (conv_param.has_pad_r()){
    pad_r_ = conv_param.pad_r();
  }
  else{
   	pad_r_ = 0;
  }
  if (conv_param.has_pad_t()){
    pad_t_ = conv_param.pad_t();
  }
  else{
    pad_t_ = 0;
  }
  if (conv_param.has_pad_b()){
    pad_b_ = conv_param.pad_b();
  }
  else{
   	pad_b_ = 0;
  }

  if(pad_l_ !=0 || pad_r_ !=0 || pad_t_ !=0 || pad_b_!=0){
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_l, pad_r, pad_t & pad_b can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_l/r/t/b should be specified; not both.";
    CHECK_EQ(0, conv_param.pad_h())
        << "Either pad_h/w or pad_l/r/t/b should be specified; not both.";
    CHECK_EQ(0, conv_param.pad_w())
        << "Either pad_h/w or pad_l/r/t/b should be specified; not both.";
    CHECK_EQ(0, conv_param.pad_type())
            << "Either pad_type or pad_l/r/t/b should be specified; not both.";
  }
  //CUSTOMIZATION-->

  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  const int* kernel_shape_data = kernel_shape_.cpu_data();
  const int* stride_data = stride_.cpu_data();
  const int* pad_data = pad_.cpu_data();
  const int* dilation_data = dilation_.cpu_data();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape[channel_axis_] *= kernel_shape_data[i];
    const int input_dim = bottom[0]->shape(channel_axis_ + i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    int output_dim;
    //<--CUSTOMIZATION
    if (pad_l_!=0 || pad_r_!=0 || pad_t_!=0 || pad_b_!=0){ //only support 2D
       if (i==0) {
         output_dim = (input_dim + pad_t_ + pad_b_ - kernel_extent) / stride_data[i] + 1;
       }
       if (i==1) {
         output_dim = (input_dim + pad_l_ + pad_r_ - kernel_extent) / stride_data[i] + 1;
       }
     }
    else{
      switch (pad_type_) {
        case 0:
    	  output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
              / stride_data[i] + 1;
    	  break;
        case 1:
    	  output_dim = ceil(float(input_dim)/float(stride_data[i]));
    	  break;
        default:
    	  LOG(FATAL) << "Unknown padding type.";
    	  break;
      }
    }
    //CUSTOMIZATION-->
    top_shape[channel_axis_ + i + 1] = output_dim;
  }
  top[0]->Reshape(top_shape);
  num_ = bottom[0]->count(0, channel_axis_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);

  channels_ = bottom[0]->shape(channel_axis_);
}

template <typename Dtype>
void Im2colLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num_; ++n) {
    DCHECK_EQ(bottom[0]->shape().size() - channel_axis_, num_spatial_axes_ + 1);
    DCHECK_EQ(top[0]->shape().size() - channel_axis_, num_spatial_axes_ + 1);
    DCHECK_EQ(kernel_shape_.count(), num_spatial_axes_);
    DCHECK_EQ(pad_.count(), num_spatial_axes_);
    DCHECK_EQ(stride_.count(), num_spatial_axes_);
    DCHECK_EQ(dilation_.count(), num_spatial_axes_);
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(bottom_data + n * bottom_dim_, channels_,
          bottom[0]->shape(channel_axis_ + 1),
          bottom[0]->shape(channel_axis_ + 2),
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
		  pad_type_, pad_l_, pad_r_, pad_t_, pad_b_, //CUSTOMIZATION
          dilation_.cpu_data()[0], dilation_.cpu_data()[1],
          top_data + n * top_dim_);
    } else {
      im2col_nd_cpu(bottom_data + n * bottom_dim_, num_spatial_axes_,
          bottom[0]->shape().data() + channel_axis_,
          top[0]->shape().data() + channel_axis_,
          kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
		  pad_type_, //CUSTOMIZATION
          dilation_.cpu_data(), top_data + n * top_dim_);
    }
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int n = 0; n < num_; ++n) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(top_diff + n * top_dim_, channels_,
          bottom[0]->shape(channel_axis_ + 1),
          bottom[0]->shape(channel_axis_ + 2),
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
		  pad_type_, pad_l_, pad_r_, pad_t_, pad_b_, //CUSTOMIZATION
          dilation_.cpu_data()[0], dilation_.cpu_data()[1],
          bottom_diff + n * bottom_dim_);
    } else {
      col2im_nd_cpu(top_diff + n * top_dim_, num_spatial_axes_,
          bottom[0]->shape().data() + channel_axis_,
          top[0]->shape().data() + channel_axis_,
          kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
		  pad_type_, //CUSTOMIZATION
          dilation_.cpu_data(), bottom_diff + n * bottom_dim_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Im2colLayer);
#endif

INSTANTIATE_CLASS(Im2colLayer);
REGISTER_LAYER_CLASS(Im2col);

}  // namespace caffe
