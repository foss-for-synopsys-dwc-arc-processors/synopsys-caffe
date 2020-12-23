#include <vector>
#include <typeinfo>
#include <math.h>

#include "caffe/layers/farthest_point_sample_layer.hpp"

namespace caffe {

template <typename Dtype>
  void FarthestPointSampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
    // first setup the params
    const PointNetParameter &point_net_param = this->layer_param_.point_net_param();
    if (point_net_param.has_n_sample_point()) {
      n_sample_point_ = point_net_param.n_sample_point();
    } else {
      n_sample_point_ = -1;
    }
    CHECK_GE(n_sample_point_, 0);
  }

  template <typename Dtype>
  void FarthestPointSampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // bottom shape = (batch_size, num_point, num_cord=3)
    // top shape = (batch_size, npoint_)
    vector<int> bottom_shape = bottom[0]->shape(), top_shape;
    CHECK_EQ(bottom_shape.size(), 3);
    bottom_shape[1] = n_sample_point_;
    bottom_shape.pop_back();
  top[0]->Reshape(bottom_shape);
  }


template <typename Dtype>
void FarthestPointSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  const int B = bottom_shape[0];
  const int N = bottom_shape[1];
  const int D = bottom_shape[2];
  vector<Dtype> dist(N);
  for(int b = 0; b < B; ++b){
    fill(dist.begin(), dist.end(), 1e38);
    top_data[0] = 0;
    int pre_far = 0; // the index of previous farthest point
    for(int k = 1; k < n_sample_point_; ++k){
      Dtype pre_x = bottom_data[pre_far * D];
      Dtype pre_y = bottom_data[pre_far * D + 1];
      Dtype pre_z = bottom_data[pre_far * D + 2];
      int max_i = -1;
      Dtype max_v = -1.0;
      for(int pt = 0; pt < N; ++pt){
        Dtype pt_x = bottom_data[pt * D];
        Dtype pt_y = bottom_data[pt * D + 1];
        Dtype pt_z = bottom_data[pt * D + 2];
        Dtype d = sqrt((pt_x-pre_x) * (pt_x-pre_x) \
                     + (pt_y-pre_y) * (pt_y-pre_y) \
                     + (pt_z-pre_z)*(pt_z-pre_z));
        if(d < dist[pt]){
          dist[pt] = d;
        }
        if(dist[pt] > max_v){
          max_v = dist[pt];
          max_i = pt;
        } else if (dist[pt] == max_v){
          // tie breaker (pointnet2): every thread first reduce over blocks (blockDim.x == 512);
          // https://github.com/charlesq34/pointnet2/blob/master/tf_ops/sampling/tf_sampling_g.cu#L130
          // then choose the largest value with smallest index
          // https://github.com/charlesq34/pointnet2/blob/master/tf_ops/sampling/tf_sampling_g.cu#L153-L165
          if( (pt & 511) < (max_i & 511) ) {
            max_i = pt;
          }
        }
      }
      pre_far = max_i;
      top_data[k] = max_i;
    }
    // stride the batch dimension
    top_data += n_sample_point_;
    bottom_data += N * D;
  }
}

INSTANTIATE_CLASS(FarthestPointSampleLayer);
REGISTER_LAYER_CLASS(FarthestPointSample);

}  // namespace caffe