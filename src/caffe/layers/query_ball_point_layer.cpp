#include <vector>
#include <typeinfo>

#include "caffe/layers/query_ball_point_layer.hpp"

namespace caffe {

template <typename Dtype>
  void QueryBallPointLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
    // first setup the params
    const PointNetParameter &point_net_param = this->layer_param_.point_net_param();
    if (point_net_param.has_n_sample_point()) {
      n_sample_point_ = point_net_param.n_sample_point();
    } else {
      n_sample_point_ = -1;
    }
    CHECK_GE(n_sample_point_, 0);
    if (point_net_param.has_radius()) {
      radius_ = point_net_param.radius();
    } else {
      radius_ = -1;
    }
    CHECK_GT(radius_, 0) << "radius should be set to be greater than zero";
  }

  template <typename Dtype>
  void QueryBallPointLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // bottom0 shape = (batch_size, num_point, num_cord=3)
    // bottom1 shape = (batch_size, npoint, num_cord=3) // npoint is from FarthestPointSample
    // top0 shape = (batch_size, npoint, nsample)
    // top1 shape = (batch_size, npoint)
    // in TF implementation it will return two outputs (idx, pts_cnt), but pts_cnt is never used in the model
    vector<int> bottom_shape = bottom[1]->shape();
    CHECK_EQ(bottom_shape.size(), 3);
    bottom_shape[2] = n_sample_point_;
    top[0]->Reshape(bottom_shape);
  }


template <typename Dtype>
void QueryBallPointLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data[2];
  bottom_data[0] = bottom[0]->cpu_data(); bottom_data[1] = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  const int B = bottom_shape[0];
  const int N = bottom_shape[1];
  const int D = bottom_shape[2];
  const int n_point = bottom[1]->shape(1);
  const float sqrRAD = radius_ * radius_;
  vector<Dtype> dist(N);
  for(int b = 0; b < B; ++b){
    for(int n = 0; n < n_point; ++n){
      const Dtype *centroid = bottom_data[1] + n * D;
      Dtype c_x = centroid[0], X;
      Dtype c_y = centroid[1], Y;
      Dtype c_z = centroid[2], Z;
      // calc distance w.r.t. bottom[0][b, :]
      const Dtype *pt = bottom_data[0];
      for(int i=0;i<N;++i){
        X = c_x - pt[0];
        Y = c_y - pt[1];
        Z = c_z - pt[2];
        dist[i] = X * X + Y * Y + Z * Z;
        pt += D;
      }
      // now dist has all the distance[centroid, all_point]
      int cnt = 0;
      int idx_0 = n * n_sample_point_;
      top_data[idx_0] = 0;
      for(int i = 0;i < N; i++){
        if(dist[i] <= sqrRAD){
          top_data[idx_0 + cnt] = i;
          ++cnt;
          if(cnt >= n_sample_point_) break;
        }
      }
      // fill with the index of first sampled point; otherwise 0
      std::fill(top_data + idx_0 + cnt, top_data + idx_0 + n_sample_point_, top_data[idx_0]);
    }
    // stride the batch dimension
    bottom_data[0] += N * D;
    bottom_data[1] += n_point * D;
    top_data += n_point * n_sample_point_;
  }
}

INSTANTIATE_CLASS(QueryBallPointLayer);
REGISTER_LAYER_CLASS(QueryBallPoint);

}  // namespace caffe