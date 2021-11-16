#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>


#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/corr1d_layer.hpp"

namespace caffe {

template <typename Dtype>
void Corr1dLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Corr1dParameter Corr1d_param1 = this->layer_param_.corr1d_param();

  num_of_shifts_ = Corr1d_param1.num_of_shifts();
  stride_        = Corr1d_param1.stride();
  LOG(INFO) << "num_of_shifts_: " <<num_of_shifts_;
  LOG(INFO) << "stride_: " <<stride_;
}

template <typename Dtype>
void Corr1dLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(4, bottom[0]->num_axes()) << "InputA must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(4, bottom[1]->num_axes()) << "InputB must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());


  LOG(INFO) << "bottom[0]->num_axes(): " <<bottom[0]->num_axes();

  top[0]->Reshape(bottom[0]->num(), num_of_shifts_/stride_, bottom[0]->height(), bottom[0]->width());
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  LOG(INFO) << "channels_: " <<channels_;
  LOG(INFO) << "height_: " <<height_;
  LOG(INFO) << "width_: " <<width_;


}

template <typename Dtype>
void Corr1dLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_dataA = bottom[0]->cpu_data();
  const Dtype* bottom_dataB = bottom[1]->cpu_data();
  Dtype* top_dataC = top[0]->mutable_cpu_data();
  //Dtype* top_dataC = top[0]->cpu_data();

  // Initialize
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_dataC);

  // The main loop
  int h1 = bottom[0]->height();
  int w1 = bottom[0]->width();
  int c1 = bottom[0]->channels();

  int h2 = top[0]->height();
  int w2 = top[0]->width();
  int c2 = top[0]->channels();
  //int out_total_len = h2 * w2 * c2;


  //printf("bottom[0].size = [%d,%d,%d] \n",c1,h1,w1);
  //printf("top[0].size = [%d,%d,%d] \n",c2,h2,w2);


  //memset(top_dataC[0], 0, out_total_len * sizeof (short));

  for (int c=0; c<c2; c++) // c is the displacement to the right
  {
    for (int y=0; y<h2; y++)
    {
        for (int x=0; x<w2; x++)
        {
            top_dataC[c*h2*w2 + y*w2 + x ] = (0);
        }
    }
  }


   Dtype sum_ch=0;

   LOG(INFO) << "fw: top[0].channels: " <<c2;
   LOG(INFO) << "fw: top[0].height:   " <<h2;
   LOG(INFO) << "fw: top[0].width:    " <<w2;




  for (int c=0; c<c2; c++) // c is the displacement to the right
  {
    for (int y=0; y<h2; y++)
    {
        for (int x=c; x<w2; x++)
        {
            sum_ch=0;
            for (int ch=0; ch<c1; ch++)
            {
                sum_ch += bottom_dataA[ch*h1*w1+ y*w1+x]*bottom_dataB[ch*h1*w1+ y*w1+x-c];
            }
            top_dataC[c*h2*w2 + y*w2 + x ] = (sum_ch);
        }
    }
  }

   LOG(INFO) << "fw: top_count:    "   << top_count ;
;
}


template <typename Dtype>
void Corr1dLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
//STUB_GPU(Corr1dLayer);
#endif

INSTANTIATE_CLASS(Corr1dLayer);
REGISTER_LAYER_CLASS(Corr1d);

}  // namespace caffe
