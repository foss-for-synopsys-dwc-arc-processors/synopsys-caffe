#include <vector>
#include <typeinfo>

#include "caffe/layers/three_interpolate_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void ThreeInterpolateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // bottom0 shape = (batch_size, num_point, num_channel)
    // bottom1 shape = (batch_size, npoint, 3)
    // bottom2 shape = (batch_size, npoint, 3) // use K=3 points to interpolate
    // top0 shape = (batch_size, npoint, num_channel)
    vector<int> bottom_shape = bottom[0]->shape();
    CHECK_EQ(bottom_shape.size(), 3);
    bottom_shape[1] = bottom[1]->shape(1);
    top[0]->Reshape(bottom_shape);
  }

template <typename Dtype>
void ThreeInterpolateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *pt0 = bottom[0]->cpu_data();
  const Dtype *idx = bottom[1]->cpu_data();
  const Dtype *weight = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  const int B = bottom_shape[0];
  const int N = bottom_shape[1];
  const int C = bottom_shape[2];
  const int n_point = bottom[1]->shape(1);
  const int K = 3; // K_interpolate
  Dtype w_1, w_2, w_3; // the weights of interpolating points
  int i_1, i_2, i_3;
  for (int b = 0; b < B; ++b) {
    for (int n = 0; n < n_point; ++n){
      i_1 = idx[n * K];
      i_2 = idx[n * K + 1];
      i_3 = idx[n * K + 2];
      w_1 = weight[n * K];
      w_2 = weight[n * K + 1];
      w_3 = weight[n * K + 2];
      for (int ch = 0; ch < C; ++ch){
        top_data[n * C + ch] = pt0[i_1 * C + ch] * w_1 \
                             + pt0[i_2 * C + ch] * w_2 \
                             + pt0[i_3 * C + ch] * w_3;
      }
    }
    // strides the batch dimension
    pt0 += N * C;
    idx += n_point * K;
    weight += n_point * K;
    top_data += n_point * C;
  }
  // refer to https://github.com/charlesq34/pointnet2/blob/master/tf_ops/3d_interpolation/interpolate.cpp#L84
}

INSTANTIATE_CLASS(ThreeInterpolateLayer);
REGISTER_LAYER_CLASS(ThreeInterpolate);

}  // namespace caffe