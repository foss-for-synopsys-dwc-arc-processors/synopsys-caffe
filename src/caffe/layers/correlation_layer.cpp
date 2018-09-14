// Modified by Synopsys Inc - Forward_cpu implementation

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/correlation_layer.hpp"

namespace caffe {

template<typename Dtype>
void blob_rearrange_kernel2(Dtype *dout, const Dtype *din, int num, int channels, int height, int width, int pheight, int pwidth, int padding)
{
    // dout[num][pwidthheight][widthheight][channels]
    // din[num][channels][height][width]

    for (int n = 0; n < num; ++n)
        for (int ch = 0; ch < channels; ++ch)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    dout[((n * pheight + y + padding) * pwidth + x + padding) * channels + ch] =
                        din[((n * channels + ch) * height + y) * width + x];
}

template <typename Dtype>
void CorrelateData(int num, int topwidth, int topheight, int topchannels, int topcount,
                   int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
                   int bottomwidth, int bottomheight, int bottomchannels,
                   const Dtype *bottom0, const Dtype *bottom1, Dtype *top)
{


    for (int n = 0; n < num; ++n)
    {
        Dtype patch_data[kernel_size * kernel_size * bottomchannels];

        for (int y = 0; y < topheight; ++y)
            for (int x = 0; x < topwidth; ++x)
            {
                int x1 = x * stride1 + max_displacement;
                int y1 = y * stride1 + max_displacement;

                // Load 3D patch into shared shared memory
                for (int j = 0; j < kernel_size; j++) // HEIGHT
                    for (int i = 0; i < kernel_size; i++) // WIDTH
                    {
                        int ji_off = ((j * kernel_size) + i) * bottomchannels;
                        for (int ch = 0; ch < bottomchannels; ch++) // CHANNELS
                        {
                            int idx1 = ((n * bottomheight + y1 + j) * bottomwidth + x1 + i) * bottomchannels + ch;
                            int idxPatchData = ji_off + ch;
                            patch_data[idxPatchData] = bottom0[idx1];
                        }
                    }

                // Compute correlation
                for (int top_channel = 0; top_channel < topchannels; top_channel++)
                {
                    Dtype sum = 0;

                    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
                    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;

                    for (int j = 0; j < kernel_size; j++) // HEIGHT
                        for (int i = 0; i < kernel_size; i++) // WIDTH
                        {
                            int ji_off = ((j * kernel_size) + i) * bottomchannels;
                            for (int ch = 0; ch < bottomchannels; ch++) // CHANNELS
                            {
                                int x2 = x1 + s2o;
                                int y2 = y1 + s2p;

                                int idxPatchData = ji_off + ch;
                                int idx2 = ((n * bottomheight + y2 + j) * bottomwidth + x2 + i) * bottomchannels + ch;

                                sum += patch_data[idxPatchData] * bottom1[idx2];
                            }
                        }




                    const int sumelems = kernel_size * kernel_size * bottomchannels;
                    const int index = ((top_channel * topheight + y) * topwidth) + x;

                    //                    printf("%f\n", sum / (float) sumelems);

                    top[index + n * topcount] = sum / (float) sumelems;
                }
            }
    }
    // Aggregate  
}

template <typename Dtype>
void CorrelateDataSubtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int topcount,
                           int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
                           int bottomwidth, int bottomheight, int bottomchannels,
                           const Dtype *bottom0, const Dtype *bottom1, Dtype *top)
{
    for (int index = 0; index < nthreads; index++)
    {
        int x = index % topwidth; //w-pos
        int y = (index / topwidth) % topheight; //h-pos
        int c = (index / topwidth / topheight) % topchannels; //channels

        // Offset of patch in image 2
        int s2o = (c % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
        int s2p = (c / neighborhood_grid_width - neighborhood_grid_radius) * stride2;

        // First (upper left) position of kernel center in current neighborhood in image 1
        int x1 = x * stride1 + kernel_radius + max_displacement;
        int y1 = y * stride1 + kernel_radius + max_displacement;

        // Iterate through 3D patch
        Dtype sum = 0;
        for (int j = -kernel_radius; j <= kernel_radius; j++)
        { // HEIGHT
            for (int i = -kernel_radius; i <= kernel_radius; i++)
            { // WIDTH
                for (int l = 0; l < bottomchannels; l++)
                { // CHANNELS
                    // Calculate position in image 2
                    int x2 = x1 + s2o;
                    int y2 = y1 + s2p;

                    // Indices in bottom data: (CH=l,W=x2,H=y2,N)
                    int idx1 = ((item * bottomheight + y1 + j) * bottomwidth + x1 + i) * bottomchannels + l;
                    int idx2 = ((item * bottomheight + y2 + j) * bottomwidth + x2 + i) * bottomchannels + l;

                    // Do the correlation:
                    sum += fabs(bottom0[idx1] - bottom1[idx2]);
                }
            }
        }
        const int sumelems = (kernel_radius * 2 + 1)*(kernel_radius * 2 + 1) * bottomchannels;
        top[index + item * topcount] = sum / (float) sumelems;
    }
}

template <typename Dtype>
void CorrelationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  CorrelationParameter corr_param = this->layer_param_.correlation_param();
  
  CHECK(corr_param.has_kernel_size()) << "Filter kernel_size is not set";
  CHECK(corr_param.has_max_displacement()) << "Max displacement is required.";
  
  kernel_size_ = corr_param.kernel_size();
  if(kernel_size_ % 2 == 0) LOG(FATAL) << "Odd kernel size required";
  
  max_displacement_ = corr_param.max_displacement();
  pad_size_ = corr_param.pad();
  stride1_ = corr_param.stride_1();
  stride2_ = corr_param.stride_2();
  
  do_abs_ = corr_param.do_abs();
  
  corr_type_ = corr_param.correlation_type();
  
  LOG(INFO) << "Kernel Size: " << kernel_size_;
  LOG(INFO) << "Stride 1: " << stride1_;
  LOG(INFO) << "Stride 2: " << stride2_;
  LOG(INFO) << "Max Displacement: " << max_displacement_;
  
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  num_ = bottom[0]->num();
  
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Both bottom blobs must have same width";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Both bottom blobs must have same height";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "Both bottom blobs must have same height";

  int bottomchannels = bottom[0]->channels();
  
  int paddedbottomheight = bottom[0]->height()+2*pad_size_;
  int paddedbottomwidth = bottom[0]->width()+2*pad_size_;  
  
  // Size computation
  kernel_radius_ = (kernel_size_ - 1) / 2; //size of unreachable border region (on each side)
  border_size_ = max_displacement_ + kernel_radius_; //size of unreachable border region (on each side)
  
  top_width_ = ceil((float)(paddedbottomwidth - border_size_*2) / (float)stride1_);
  top_height_ = ceil((float)(paddedbottomheight - border_size_*2) / (float)stride1_);

  CHECK_GE(top_width_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  CHECK_GE(top_height_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  
  // Given a center position in image 1, how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
  neighborhood_grid_radius_ = max_displacement_ / stride2_;
  neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

  // Top Channels amount to displacement combinations in X and Y direction:
  top_channels_ = neighborhood_grid_width_ * neighborhood_grid_width_;
  
  //Reshape top
  top[0]->Reshape(num_, top_channels_, top_height_, top_width_);

  // rbots (These are the blobs that store the padded and dimension rearranged data
  rbot1_.reset(new Blob<Dtype>());
  rbot2_.reset(new Blob<Dtype>());
  rbot1_->Reshape(num_, paddedbottomheight, paddedbottomwidth, bottomchannels);
  rbot2_->Reshape(num_, paddedbottomheight, paddedbottomwidth, bottomchannels);
  
  rtopdiff_.reset(new Blob<Dtype>());
  rtopdiff_->Reshape(num_, top_height_, top_width_, top_channels_);

}

template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    CHECK_EQ(bottom.size(), 2);
    CHECK_EQ(top.size(), 1);

    const int bnum = bottom[0]->num();
    const int bchannels = bottom[0]->channels();
    const int bheight = bottom[0]->height();
    const int bwidth = bottom[0]->width();
    
    const int topcount = top_width_ * top_height_ * top_channels_;
    const int pheight = bheight + 2 * pad_size_;
    const int pwidth = bwidth + 2 * pad_size_;
    
    memset(rbot1_->mutable_cpu_data(), 0, sizeof(Dtype) * rbot1_->count());
    memset(rbot2_->mutable_cpu_data(), 0, sizeof(Dtype) * rbot2_->count());

    blob_rearrange_kernel2<Dtype>(rbot1_->mutable_cpu_data(), bottom[0]->cpu_data(), bnum, bchannels, bheight, bwidth, pheight, pwidth, pad_size_);
    blob_rearrange_kernel2<Dtype>(rbot2_->mutable_cpu_data(), bottom[1]->cpu_data(), bnum, bchannels, bheight, bwidth, pheight, pwidth, pad_size_);

    const int num = bnum;
    const int channels = bchannels;
    const int height = bheight + 2 * pad_size_;
    const int width = bwidth + 2 * pad_size_;
    
    if(corr_type_ == CorrelationParameter_CorrelationType_MULTIPLY) {
        // CorrelationLayer      
        
        CorrelateData<Dtype>(
            num, top_width_, top_height_, top_channels_, topcount,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_, kernel_size_,
            stride1_, stride2_,
            width, height, channels,
            rbot1_->cpu_data(), rbot2_->cpu_data(), top[0]->mutable_cpu_data()
            );
        
    } else if(corr_type_ == CorrelationParameter_CorrelationType_SUBTRACT) {
        // CorrelationLayer
        for(int n = 0; n < num; n++) {
            
            int topThreadCount = topcount;
            CorrelateDataSubtract<Dtype>(
                topThreadCount, num, n, top_width_, top_height_, top_channels_, topcount,
                max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
                stride1_, stride2_,
                width, height, channels,
                rbot1_->cpu_data(), rbot2_->cpu_data(), top[0]->mutable_cpu_data()
                );
        }
    }
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CorrelationLayer);
#endif

INSTANTIATE_CLASS(CorrelationLayer);
REGISTER_LAYER_CLASS(Correlation);

}  // namespace caffe
