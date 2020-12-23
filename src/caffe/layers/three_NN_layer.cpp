#include <vector>
#include <typeinfo>

#include "caffe/layers/three_NN_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void ThreeNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // bottom0 shape = (batch_size, num_point, num_cord=3)
    // bottom1 shape = (batch_size, npoint, num_cord=3)
    // top0 shape = (batch_size, num_point, num_cord=3): dist
    // top1 shape = (batch_size, num_point, num_cord=3): index
    vector<int> bottom_shape = bottom[0]->shape();
    CHECK_EQ(bottom_shape.size(), 3);
    top[0]->Reshape(bottom_shape);
    top[1]->Reshape(bottom_shape);
  }

template <typename Dtype>
void ThreeNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *pt0 = bottom[0]->cpu_data();
  const Dtype *pt1 = bottom[1]->cpu_data();
  Dtype* dist = top[0]->mutable_cpu_data();
  Dtype* idx = top[1]->mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  const int B = bottom_shape[0];
  const int N = bottom_shape[1];
  const int D = bottom_shape[2];
  const int n_point = bottom[1]->shape(1);
  const int K = 3;
  for (int b = 0; b < B; ++b){
    for (int n = 0; n < N; ++n) {
      Dtype x0 = pt0[n * D + 0];
      Dtype y0 = pt0[n * D + 1];
      Dtype z0 = pt0[n * D + 2];
      Dtype best_v0 = 1e40; Dtype best_v1 = 1e40; Dtype best_v2 = 1e40;
      int best_i0 = 0; int best_i1 = 0; int best_i2 = 0;
      for (int m = 0; m < n_point; ++m) {
        Dtype x1 = x0 - pt1[m * D + 0];
        Dtype y1 = y0 - pt1[m * D + 1];
        Dtype z1 = z0 - pt1[m * D + 2];
        Dtype tmp_d = x1*x1 + y1*y1 + z1*z1;
        if (tmp_d < best_v0) {
          best_v2 = best_v1;
          best_i2 = best_i1;
          best_v1 = best_v0;
          best_i1 = best_i0;
          best_v0 = tmp_d;
          best_i0 = m;
        } else if (tmp_d < best_v1) {
          best_v2 = best_v1;
          best_i2 = best_i1;
          best_v1 = tmp_d;
          best_i1 = m;
        } else if (tmp_d < best_v2) {
          best_v2 = tmp_d;
          best_i2 = m;
        }
      }
      dist[n * K] = best_v0;
      dist[n * K + 1] = best_v1;
      dist[n * K + 2] = best_v2;
      idx[n * K] = best_i0;
      idx[n * K + 1] = best_i1;
      idx[n * K + 2] = best_i2;
    }
    pt0 += N * D;
    pt1 += n_point * D;
    dist += N * K;
    idx += N * K;
  }
  // refer to https://github.com/charlesq34/pointnet2/blob/master/tf_ops/3d_interpolation/interpolate.cpp#L21
  // k-NN might be used as a substitue of QueryBallPoint, refer to https://github.com/charlesq34/pointnet2/blob/master/tf_ops/grouping/tf_grouping.py#L48-L73
  // where tf.nn.top_k (synopsys_caffe::ArgMax) should be considered, rather than modifying this layer
}

INSTANTIATE_CLASS(ThreeNNLayer);
REGISTER_LAYER_CLASS(ThreeNN);

}  // namespace caffe