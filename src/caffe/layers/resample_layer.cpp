// Copyright 2014 BVLC and contributors.
// Modified by Synopsys Inc - Forward_cpu implementation

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/resample_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define min(a,b) ((a<b)?a:b)

namespace caffe {

static float bicubicCoeff(float x_)
{
    float x = fabs(x_);
    if (x <= 1.0f)     return x * x * (1.5f * x - 2.5f) + 1.0f;
    else if (x < 2.0f) return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
    else               return 0.0f;
}

static float boxCoeff(float x)
{
    if (-0.5 <= x  && x<0.5) return 1.0;
    return 0;
}

static float triangleCoeff(float x)
{
    if (-1<=x && x<0) return x+1;
    if (0<=x && x<=1) return 1-x;
    return 0;
}

#define FILTER_BICUBIC 0
#define FILTER_BOX 1
#define FILTER_TRIANGLE 2

template <typename Dtype>
void InterpolationKernel(
                         const int nthreads,
                         const int in_channelsize,
                         const int out_channelsize,
                         const Dtype* in_ptr,
                         const int in_width,
                         const int in_height,
                         const float fx,
                         const float fy,
                         Dtype* out_ptr,
                         const int out_width,
                         const int out_height,
                         int filter_type,
                         int kernel_width,
                         const bool antialias)
{
    for (int index = 0; index < nthreads; ++index)
    {
        int c = index / out_channelsize;
        int x_out = (index % out_channelsize) % out_width;
        int y_out = (index % out_channelsize) / out_width;

        float x_in = x_out * fx + fy / 2.0f - 0.5f;
        float y_in = y_out * fy + fx / 2.0f - 0.5f;

        int x_in_round = round(x_in);
        int y_in_round = round(y_in);

        Dtype sum = 0;
        Dtype wsum = 0;

        float ax = 1.0f / (antialias ? fx : 1.0f);
        float ay = 1.0f / (antialias ? fy : 1.0f);
        int rx = (fx < 1.0f) ? 2 : ceil(float(kernel_width) / ax);
        int ry = (fy < 1.0f) ? 2 : ceil(float(kernel_width) / ay);

        for (int y = y_in_round - ry; y <= y_in_round + ry; y++)
            for (int x = x_in_round - rx; x <= x_in_round + rx; x++)
            {
                if (y < 0 || x < 0) continue;
                if (y >= in_height || x >= in_width) continue;

                float dx = x_in - x;
                float dy = y_in - y;

                float w;
                if (filter_type == FILTER_BICUBIC) w = ax * bicubicCoeff(ax * dx) * ay * bicubicCoeff(ay * dy);
                else if (filter_type == FILTER_BOX) w = ax * boxCoeff(ax * dx) * ay * boxCoeff(ay * dy);
                else w = ax * triangleCoeff(ax * dx) * ay * triangleCoeff(ay * dy);
                sum += w * in_ptr[c * in_channelsize + y * in_width + x];
                wsum += w;
            }

        out_ptr[index] = (!wsum) ? 0 : (sum / wsum);
    }
}

template <typename Dtype>
void NearestNeighborKernel(
                           const int nthreads,
                           const int in_channelsize,
                           const int out_channelsize,
                           const Dtype* in_ptr,
                           const int in_width,
                           const int in_height,
                           const float fx,
                           const float fy,
                           Dtype* out_ptr,
                           const int out_width,
                           const int out_height)
{
    for (int index = 0; index < nthreads; ++index)
    {
        int c = index / out_channelsize;
        int x_out = (index % out_channelsize) % out_width;
        int y_out = (index % out_channelsize) / out_width;

        float x_in = x_out * fx + fy / 2.0f - 0.5f;
        float y_in = y_out * fy + fx / 2.0f - 0.5f;

        int x_in_round = round(x_in);
        int y_in_round = round(y_in);

        out_ptr[index] = in_ptr[c * in_channelsize + y_in_round * in_width + x_in_round];
    }
}

template <typename Dtype>
void ResampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    if(this->layer_param().resample_param().type() != ResampleParameter_ResampleType_CUBIC
            && this->layer_param().resample_param().type() != ResampleParameter_ResampleType_LINEAR
            && this->layer_param().resample_param().type() != ResampleParameter_ResampleType_NEAREST)
        LOG(FATAL) << "ResampleLayer: only CUBIC, LINEAR and NEAREST interpolation is supported for now";
}

template <typename Dtype>
void ResampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "ResampleLayer only runs Reshape on setup";
  
  CHECK_GE(bottom.size(), 1);
  CHECK_LE(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();

  int top_width;
  int top_height;

  if (bottom.size() == 1) {
    top_height = this->layer_param_.resample_param().height();
    top_width = this->layer_param_.resample_param().width();
  } else {
    top_height = bottom[1]->height();
    top_width = bottom[1]->width();
  }

  CHECK_GE(top_height, 1) << "ResampleLayer must have top_height > 0";
  CHECK_GE(top_width, 1) << "ResampleLayer must have top_width > 0";

  int count = top_width * top_height * channels * num;
  
  top[0]->Reshape(num, channels, top_height, top_width);
  CHECK_EQ(count, top[0]->count());
}

template <typename Dtype>
void ResampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
{

    Dtype* top_data = top[0]->mutable_cpu_data(); // dest
    int topwidth = top[0]->width();
    int topheight = top[0]->height();
    int topchannels = top[0]->channels();
//    int topcount = top[0]->count();

    Dtype* bottom_data = bottom[0]->mutable_cpu_data(); // source
    int bottomnum = (bottom)[0]->num();
    int bottomchannels = (bottom)[0]->channels();
    int bottomwidth = (bottom)[0]->width();
    int bottomheight = (bottom)[0]->height();
//    int bottomcount = (bottom)[0]->count();

    CHECK_EQ(topchannels, bottomchannels) << "ResampleLayer top channel count must match bottom channel count";

    float fx = float(bottomwidth) / float(topwidth);
    float fy = float(bottomheight) / float(topheight);

    //int botsize = bottomwidth*bottomheight*bottomchannels*bottomnum;
    int topsize = topwidth * topheight * topchannels*bottomnum;
    int topchannelsize = topwidth*topheight;
    int botchannelsize = bottomwidth*bottomheight;

    if (this->layer_param().resample_param().type() == ResampleParameter_ResampleType_NEAREST)
    {
        NearestNeighborKernel<Dtype>(
                topsize,
                botchannelsize,
                topchannelsize,
                (Dtype*) bottom_data,
                bottomwidth,
                bottomheight,
                fx,
                fy,
                (Dtype*) top_data,
                topwidth,
                topheight
                );
    }
    else if (this->layer_param().resample_param().type() == ResampleParameter_ResampleType_CUBIC || this->layer_param().resample_param().type() == ResampleParameter_ResampleType_LINEAR)
    {
        int filter_type;
        if (this->layer_param().resample_param().type() == ResampleParameter_ResampleType_CUBIC)
            filter_type = FILTER_BICUBIC;
        else if (this->layer_param().resample_param().type() == ResampleParameter_ResampleType_LINEAR)
            filter_type = FILTER_TRIANGLE;

        bool isDownsample = (fx > 1) || (fy > 1);
        bool antialias = isDownsample && this->layer_param_.resample_param().antialias();

        int kernel_width;
        if (filter_type == FILTER_BICUBIC) kernel_width = 4;
        else if (filter_type == FILTER_BOX) kernel_width = 1;
        else kernel_width = 2;

        InterpolationKernel<Dtype>(
                topsize,
                botchannelsize,
                topchannelsize,
                (Dtype*) bottom_data,
                bottomwidth,
                bottomheight,
                fx,
                fy,
                (Dtype*) top_data,
                topwidth,
                topheight,
                filter_type,
                kernel_width,
                antialias);
    }
    else
        LOG(FATAL) << "unsupported downsampling type";
}

template <typename Dtype>
void ResampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  LOG(FATAL) << "ResampleLayer cannot do backward.";
}

#ifdef CPU_ONLY
STUB_GPU(ResampleLayer);
#endif

INSTANTIATE_CLASS(ResampleLayer);
REGISTER_LAYER_CLASS(Resample);

}  // namespace caffe
