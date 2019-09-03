#include <algorithm>
#include <vector>

#include "caffe/layers/strided_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StridedSliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  const StridedSliceParameter &strided_slice_param =
      this->layer_param_.strided_slice_param();
  strided_begin_.clear();
  std::copy(strided_slice_param.begin().begin(),
            strided_slice_param.begin().end(),
            std::back_inserter(strided_begin_));
  strided_end_.clear();
  std::copy(strided_slice_param.end().begin(), strided_slice_param.end().end(),
            std::back_inserter(strided_end_));
  strides_.clear();
  std::copy(strided_slice_param.strides().begin(),
            strided_slice_param.strides().end(), std::back_inserter(strides_));
  CHECK_EQ(strided_begin_.size(), strided_end_.size())
      << "begin, end and strides should have the same length n";
  if (strides_.size() != 0) {
    CHECK_EQ(strided_begin_.size(), strides_.size())
        << "begin, end and strides should have the same length n";
    for (int i = 0; i < strides_.size(); ++i) {
      CHECK_NE(strides_[i], 0) << "strides should be non-zero!";
    }
  } else {
    strides_.assign(strided_begin_.size(), 1);
  }

  // add onnx operator Slice parameter: axes
  axes_.clear();
  std::copy(strided_slice_param.axes().begin(),
            strided_slice_param.axes().end(), std::back_inserter(axes_));
  if (axes_.size() != 0) {
    onnx_flag_ = 1;
    CHECK_EQ(axes_.size(), strided_begin_.size())
        << "axes and starts should have the same length!";
    // make axes_ not negative
    for (int i = 0; i < axes_.size(); ++i) {
      axes_[i] = (axes_[i] < 0) ? axes_[i] + bottom[0]->num_axes() : axes_[i];
    }
  } else {
    onnx_flag_ = 0;
  }

  b_shape = bottom[0]->shape();
  s_len_ = strided_begin_.size();

  if (onnx_flag_ == 0) {
    // get mask parameter
    begin_mask_ = strided_slice_param.begin_mask();
    end_mask_ = strided_slice_param.end_mask();
    new_axis_mask_ = strided_slice_param.new_axis_mask();
    shrink_axis_mask_ = strided_slice_param.shrink_axis_mask();
    ellipsis_mask_ = strided_slice_param.ellipsis_mask();
    CHECK_EQ(ellipsis_mask_ & (ellipsis_mask_ - 1), 0)
        << "Only one non-zero bit is allowed in ellipsis_mask! ";
    // shrink_axis_mask_ , new_axis_mask_
    int ellipsis_axis = -1;
    for (int i = 0; i < s_len_; ++i) {
      if (ellipsis_mask_ & 1 << i) {
        ellipsis_axis = i;
      }
      if (shrink_axis_mask_ & 1 << i && i != ellipsis_axis) {
        strided_end_[i] = strided_begin_[i] + strides_[i] / abs(strides_[i]);
        strides_[i] = strides_[i] / abs(strides_[i]);
      }
      if (new_axis_mask_ & 1 << i && i != ellipsis_axis) {
        if (ellipsis_axis == -1 || i < ellipsis_axis) {
          b_shape.insert(b_shape.begin() + i, 1);
          strided_begin_[i] = 0;
          strided_end_[i] = 1;
          strides_[i] = 1;
        }
      }
    }
    for (int i = s_len_ - 1; i > -1; --i) {
      if (new_axis_mask_ & 1 << i && i > ellipsis_axis && ellipsis_axis != -1) {
        b_shape.insert(b_shape.end() + i - s_len_ + 1, 1);
        strided_begin_[i] = 0;
        strided_end_[i] = 1;
        strides_[i] = 1;
      }
    }
    // ellipsis_mask_
    const int b_dim = b_shape.size();
    const int add_n = b_dim - s_len_;
    strided_begin_.resize(b_dim);
    strided_end_.resize(b_dim);
    strides_.resize(b_dim);
    if (ellipsis_axis != -1) {
      for (int i = s_len_ - 1; i > ellipsis_axis; --i) {
        strided_begin_[i + add_n] = strided_begin_[i];
        strided_end_[i + add_n] = strided_end_[i];
        strides_[i + add_n] = strides_[i];
      }
      for (int i = ellipsis_axis; i < ellipsis_axis + add_n + 1; i++) {
        strided_begin_[i] = 0;
        strided_end_[i] = b_shape[i];
        strides_[i] = 1;
      }
    } else {
      if (add_n > 0) {
        for (int i = b_dim - add_n; i < b_dim; ++i) {
          strided_begin_[i] = 0;
          strided_end_[i] = b_shape[i];
          strides_[i] = 1;
        }
      }
    }
    // make sure all elements in begin, end, strides are not negative
    for (int i = 0; i < b_dim; ++i) {
      if (strided_begin_[i] >= 0) {
        strided_begin_[i] = std::min(strided_begin_[i], b_shape[i] - 1);
      } else {
        strided_begin_[i] = std::max(strided_begin_[i] + b_shape[i], 0);
      }
      if (strided_end_[i] >= 0) {
        strided_end_[i] = std::min(strided_end_[i], b_shape[i]);
      } else {
        strided_end_[i] = std::max(strided_end_[i] + b_shape[i], 0);
      }
    }
    // begin_mask and end_mask
    for (int i = 0; i < s_len_; ++i) {
      if ((begin_mask_ & 1 << i) && (ellipsis_axis != i) &&
          !(shrink_axis_mask_ & 1 << i) && !(new_axis_mask_ & 1 << i)) {
        if (ellipsis_axis != -1 && i > ellipsis_axis) {
          strided_begin_[i + add_n] =
              (strides_[i + add_n] > 0) ? 0 : b_shape[i + add_n] - 1;
        } else {
          strided_begin_[i] = (strides_[i] > 0) ? 0 : b_shape[i] - 1;
        }
      }
      if ((end_mask_ & 1 << i) && (ellipsis_axis != i) &&
          !(shrink_axis_mask_ & 1 << i) && !(new_axis_mask_ & 1 << i)) {
        if (ellipsis_axis != -1 && i > ellipsis_axis) {
          strided_end_[i + add_n] =
              (strides_[i + add_n] > 0) ? b_shape[i + add_n] : -1;
        } else {
          strided_end_[i] = (strides_[i] > 0) ? b_shape[i] : -1;
        }
      }
    }
    // t_shape
    t_shape = b_shape;
    for (int i = 0; i < b_shape.size(); ++i) {
      if (strides_[i] > 0) {
        t_shape[i] =
            (strided_end_[i] - strided_begin_[i] - 1) / strides_[i] + 1;
      } else {
        t_shape[i] =
            (strided_end_[i] - strided_begin_[i] + 1) / strides_[i] + 1;
      }
    }
    t_shape2 = t_shape;
    for (int i = s_len_ - 1; i > -1; --i) {
      if (new_axis_mask_ & 1 << i) {
      } else {
        if (shrink_axis_mask_ & 1 << i) {
          if (ellipsis_mask_ != 0 && i > ellipsis_axis) {
            t_shape2.erase(t_shape2.begin() + i + add_n);
          }
          if (ellipsis_mask_ == 0 || i < ellipsis_axis) {
            t_shape2.erase(t_shape2.begin() + i);
          }
        }
      }
    }
  }
  // onnx parameter initialization
  else {
    t_shape = b_shape;
    for (int i = 0; i < axes_.size(); ++i) {
      // make sure all elements in begin, end, strides are in bound [0,
      // bottom_shape(axes_[i])]
      if (strided_begin_[i] >= 0) {
        strided_begin_[i] = std::min(strided_begin_[i], b_shape[axes_[i]] - 1);
      } else {
        strided_begin_[i] = std::max(strided_begin_[i] + b_shape[axes_[i]], 0);
      }
      if (strided_end_[i] >= 0) {
        strided_end_[i] = std::min(strided_end_[i], b_shape[axes_[i]]);
      } else {
        strided_end_[i] = std::max(strided_end_[i] + b_shape[axes_[i]], 0);
      }
      // caculate top shape
      if (strides_[i] > 0) {
        t_shape[axes_[i]] =
            (strided_end_[i] - strided_begin_[i] - 1) / strides_[i] + 1;
      } else {
        t_shape[axes_[i]] =
            (strided_end_[i] - strided_begin_[i] + 1) / strides_[i] + 1;
      }
    }
    // complete in all dimensions
    if (s_len_ < b_shape.size()) {
      vector<int> starts = strided_begin_;
      vector<int> ends = strided_end_;
      vector<int> steps = strides_;
      strided_begin_.assign(b_shape.size(), 0);
      strided_end_ = b_shape;
      strides_.assign(b_shape.size(), 1);
      for (int i = 0; i < axes_.size(); ++i) {
        strided_begin_[axes_[i]] = starts[i];
        strided_end_[axes_[i]] = ends[i];
        strides_[axes_[i]] = steps[i];
      }
    }
  }
}

template <typename Dtype>
void StridedSliceLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  if (onnx_flag_ == 0) {
    top[0]->Reshape(t_shape2);
  } else {
    top[0]->Reshape(t_shape);
  }
}

template <typename Dtype>
inline vector<int>
StridedSliceLayer<Dtype>::strided_slice(const vector<int> &t_shape,
                                        const vector<int> &begin,
                                        const vector<int> &strides) const {
  std::vector<int> strided_idx;
  for (int i = 0; i < begin.size(); ++i) {
    for (int j = 0; j < t_shape[i]; ++j) {
      strided_idx.insert(strided_idx.end(), begin[i] + j * strides[i]);
    }
  }
  return strided_idx;
}

template <typename Dtype>
inline vector<int>
StridedSliceLayer<Dtype>::Indices(int offset, const vector<int> &top_shape,
                                  const vector<int> &strided_idx) const {
  vector<int> indices0(top_shape.size(), 0);
  vector<int> indices1(top_shape.size(), 0);
  int r = offset;
  int ts = 0;
  for (int i = top_shape.size() - 1; i >= 0; i--) {
    indices0[i] = r % top_shape[i];
    r /= top_shape[i];
  }
  for (int i = 0; i < top_shape.size(); ++i) {
    indices1[i] = strided_idx[ts + indices0[i]];
    ts = ts + top_shape[i];
  }
  return indices1;
}

template <typename Dtype>
inline int StridedSliceLayer<Dtype>::offset(const vector<int> &b_shape,
                                            const vector<int> &indices) const {
  int offset = 0;
  for (int i = 0; i < b_shape.size(); ++i) {
    int count_shape = 1;
    for (int j = i + 1; j < b_shape.size(); ++j) {
      count_shape *= b_shape[j];
    }
    offset += indices[i] * count_shape;
  }
  return offset;
}

template <typename Dtype>
void StridedSliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  std::vector<int> strided_idx =
      strided_slice(t_shape, strided_begin_, strides_);
  for (int i = 0; i < top[0]->count(); ++i) {
    vector<int> indices = Indices(i, t_shape, strided_idx);
    int b_idx = offset(b_shape, indices);
    top_data[i] = bottom_data[b_idx];
  }
}

INSTANTIATE_CLASS(StridedSliceLayer);
REGISTER_LAYER_CLASS(StridedSlice);

} // namespace caffe
