#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/mul_layer.hpp"

namespace caffe {

template <typename Dtype>
void MulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  const MulParameter &mul_param = this->layer_param_.mul_param();
  mul_shape.clear();
  std::copy(mul_param.mul_shape().begin(), mul_param.mul_shape().end(),
            std::back_inserter(mul_shape));
  if (bottom.size() > 1) {
    multiplier = bottom[1];
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(mul_shape));
    multiplier = this->blobs_[0].get();
  }
}

template <typename Dtype>
void MulLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top) {
  CHECK_NE(top[0], bottom[0]) << this->type()
                              << " Layer does not allow in-place computation.";
  is_scalar_ = false;

  if (bottom[0]->num_axes() == 0 || multiplier->num_axes() == 0)
    is_scalar_ = true;
  dim_diff_ = bottom[0]->num_axes() - multiplier->num_axes();
  dim_ = bottom[0]->num_axes() >= multiplier->num_axes()
             ? bottom[0]->num_axes()
             : multiplier->num_axes();
  vector<int> top_shape(dim_, 1);
  if (dim_diff_ == 0) {
    if (!is_scalar_) {
      for (int i = 0; i < dim_; i++) {
        CHECK(bottom[0]->shape(i) == multiplier->shape(i) ||
              bottom[0]->shape(i) == 1 || multiplier->shape(i) == 1)
            << "Dimensions must be equal or 1 in the bottoms!";
        top_shape[i] = bottom[0]->shape(i) >= multiplier->shape(i)
                           ? bottom[0]->shape(i)
                           : multiplier->shape(i);
      }
    }
  } else if (dim_diff_ > 0) // bottom0 has more axes than bottom1
  {
    if (!is_scalar_) {
      for (int i = 0; i < dim_diff_; i++)
        top_shape[i] = bottom[0]->shape(i);
      for (int i = dim_diff_; i < dim_; i++)
        top_shape[i] = bottom[0]->shape(i) >= multiplier->shape(i - dim_diff_)
                           ? bottom[0]->shape(i)
                           : multiplier->shape(i - dim_diff_);
    } else // bottom1 is a scalar
    {
      for (int i = 0; i < dim_; i++)
        top_shape[i] = bottom[0]->shape(i);
    }
  } else // dim_diff_<0, bottom1 has more axes than bottom0
  {
    if (!is_scalar_) {
      for (int i = 0; i < -dim_diff_; i++)
        top_shape[i] = multiplier->shape(i);
      for (int i = -dim_diff_; i < dim_; i++)
        top_shape[i] = bottom[0]->shape(i + dim_diff_) >= multiplier->shape(i)
                           ? bottom[0]->shape(i + dim_diff_)
                           : multiplier->shape(i);
    } else // bottom0 is a scalar
    {
      for (int i = 0; i < dim_; i++)
        top_shape[i] = multiplier->shape(i);
    }
  }

  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom0_data = bottom[0]->cpu_data();
  const Dtype *bottom1_data = multiplier->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  int count = top[0]->count();
  // Assume top index (x,y,z) with top shape (A, B, C)
  // top offset d = xBC + yC + z
  // So to count the bottom index, should first figure out x, y, z
  // x = d / BC
  // y = (d % BC) / C
  // z = d % C
  // Then consider bottom shape (A', B', C'), while A = A' or 1
  // So bottom offset = x'B'C' + y'C' + z', while x' = x or 0
  if (!is_scalar_) {
    for (int d = 0; d < count; d++) {
      int offset0 = 0;
      int offset1 = 0;

      if (dim_diff_ == 0) {
        for (int i = 0; i < dim_ - 1; i++) {
          int num = (d % top[0]->count(i)) / top[0]->count(i + 1);
          int n0 = 1 == bottom[0]->shape(i) ? 0 : num;
          int n1 = 1 == multiplier->shape(i) ? 0 : num;
          offset0 += n0 * bottom[0]->count(i + 1);
          offset1 += n1 * multiplier->count(i + 1);
        }
        int z = d % top[0]->shape(dim_ - 1);
        int z0 = 1 == bottom[0]->shape(dim_ - 1) ? 0 : z;
        int z1 = 1 == multiplier->shape(dim_ - 1) ? 0 : z;
        offset0 += z0;
        offset1 += z1;
      } else if (dim_diff_ > 0) // bottom0 has more axes than bottom1
      {
        for (int i = 0; i < dim_diff_; i++) {
          int num = (d % top[0]->count(i)) / top[0]->count(i + 1);
          int n0 = 1 == bottom[0]->shape(i) ? 0 : num;
          offset0 += n0 * bottom[0]->count(i + 1);
        }
        for (int i = dim_diff_; i < dim_ - 1; i++) {
          int num = (d % top[0]->count(i)) / top[0]->count(i + 1);
          int n0 = 1 == bottom[0]->shape(i) ? 0 : num;
          int n1 = 1 == multiplier->shape(i - dim_diff_) ? 0 : num;
          offset0 += n0 * bottom[0]->count(i + 1);
          offset1 += n1 * multiplier->count(i - dim_diff_ + 1);
        }
        int z = d % top[0]->shape(dim_ - 1);
        int z0 = 1 == bottom[0]->shape(dim_ - 1) ? 0 : z;
        int z1 = 1 == multiplier->shape(dim_ - dim_diff_ - 1) ? 0 : z;
        offset0 += z0;
        offset1 += z1;
      } else // dim_diff_<0, bottom1 has more axes than bottom0
      {
        for (int i = 0; i < -dim_diff_; i++) {
          int num = (d % top[0]->count(i)) / top[0]->count(i + 1);
          int n1 = 1 == multiplier->shape(i) ? 0 : num;
          offset1 += n1 * multiplier->count(i + 1);
        }
        for (int i = -dim_diff_; i < dim_ - 1; i++) {
          int num = (d % top[0]->count(i)) / top[0]->count(i + 1);
          int n0 = 1 == bottom[0]->shape(i + dim_diff_) ? 0 : num;
          int n1 = 1 == multiplier->shape(i) ? 0 : num;
          offset0 += n0 * bottom[0]->count(i + dim_diff_ + 1);
          offset1 += n1 * multiplier->count(i + 1);
        }
        int z = d % top[0]->shape(dim_ - 1);
        int z0 = 1 == bottom[0]->shape(dim_ + dim_diff_ - 1) ? 0 : z;
        int z1 = 1 == multiplier->shape(dim_ - 1) ? 0 : z;
        offset0 += z0;
        offset1 += z1;
      }

      top_data[d] = bottom0_data[offset0] * bottom1_data[offset1];
    }
  } else // is scalar with shape ()
  {
    if (multiplier->num_axes() == 0) // bottom1 is a scalar
    {
      Dtype scalar = bottom1_data[0];
      caffe_cpu_scale(count, scalar, bottom0_data, top_data);
    } else // bottom0 is a scalar
    {
      Dtype scalar = bottom0_data[0];
      caffe_cpu_scale(count, scalar, bottom1_data, top_data);
    }
  }
}

INSTANTIATE_CLASS(MulLayer);
REGISTER_LAYER_CLASS(Mul);

} // namespace caffe
