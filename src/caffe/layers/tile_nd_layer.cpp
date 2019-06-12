#include <vector>

#include "caffe/layers/tile_nd_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TileNDLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const TileNDParameter& tile_nd_param = this->layer_param_.tile_nd_param();
  axis_.clear();
  std::copy(tile_nd_param.axis().begin(),
    tile_nd_param.axis().end(),
    std::back_inserter(axis_));
  tiles_.clear();
  std::copy(tile_nd_param.tiles().begin(),
    tile_nd_param.tiles().end(),
    std::back_inserter(tiles_));

  CHECK_EQ(axis_.size(), tiles_.size()) << "Number of tiles must be equal to axis!";
  CHECK_GT(tiles_.size(), 0) << "Number of tiles must be positive!";
  for(int i=0;i<axis_.size();i++)
  {
    axis_[i] = bottom[0]->CanonicalAxisIndex(axis_[i]);
    CHECK_GT(tiles_[i], 0) << "Value of tiles must be positive!";
  }
}


template <typename Dtype>
void TileNDLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  for(int i=axis_.size()-1;i>=0;i--)
    top_shape[axis_[i]] = bottom[0]->shape(axis_[i]) * tiles_[i];

  top[0]->Reshape(top_shape);
  top_temp_.Reshape(top_shape);

  outer_dim_.clear();
  inner_dim_.clear();

  int multiple = 1;
  for(int i=axis_.size()-1; i>=0; i--) //notice the order
  {
    outer_dim_.push_back(bottom[0]->count(0, axis_[i]));
    inner_dim_.push_back(bottom[0]->count(axis_[i]));
    top_inner_dim_.push_back(bottom[0]->count(axis_[i])*multiple);
    multiple *= tiles_[i];
  }

  // reverse the order to make it align with the tiles and handling (from right to left dimension)
  vector<int> tmp(outer_dim_);
  outer_dim_.clear();
  std::copy(tmp.rbegin(), tmp.rend(), std::back_inserter(outer_dim_));
  tmp = inner_dim_;
  inner_dim_.clear();
  std::copy(tmp.rbegin(), tmp.rend(), std::back_inserter(inner_dim_));
  tmp = top_inner_dim_;
  top_inner_dim_.clear();
  std::copy(tmp.rbegin(), tmp.rend(), std::back_inserter(top_inner_dim_));

  //for(int i=0;i<axis_.size();i++)
  //  LOG(INFO)<<"out: "<<outer_dim_[i]<<", in: "<<inner_dim_[i]<<", top: "<<top_inner_dim_[i]<<"\n";
}

template <typename Dtype>
void TileNDLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //Dtype* top_data = top[0]->mutable_cpu_data();
  int s = outer_dim_.size()-1; //copy from most right dimension
  Dtype* top_data_temp = top_temp_.mutable_cpu_data();
  for (int i = 0; i < outer_dim_[s]; ++i) {
    for (int t = 0; t < tiles_[s]; ++t) {
      caffe_copy(inner_dim_[s], bottom_data, top_data_temp);
      top_data_temp += inner_dim_[s];
    }
    bottom_data += inner_dim_[s];
  }

  //top_data_temp = top_temp_.mutable_cpu_data();
  //for (int i = 0; i < top_temp_.count(); ++i)
  //  LOG(INFO)<<top_data_temp[i]<<"\n";
  //LOG(INFO)<<"stage2\n";

  for(int s=outer_dim_.size()-2;s>0;s--)
  {
    Dtype* input_data = top_temp_.mutable_cpu_data();
    Dtype* output_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < outer_dim_[s]; ++i) {
      for (int t = 0; t < tiles_[s]; ++t) {
        caffe_copy(top_inner_dim_[s], input_data, output_data);
        output_data += top_inner_dim_[s];
      }
      input_data += top_inner_dim_[s];
    }

    //top_data_temp = top[0]->mutable_cpu_data();
    //for (int i = 0; i < top[0]->count(); ++i)
    //  LOG(INFO)<<top_data_temp[i]<<"\n";
    //LOG(INFO)<<"stage3\n";

    Dtype* next_input_data = top[0]->mutable_cpu_data();
    Dtype* copy_data = top_temp_.mutable_cpu_data();
    caffe_copy(top[0]->count(axis_[s-1]), next_input_data, copy_data); //reset for the next loop

    //top_data_temp = top_temp_.mutable_cpu_data();
    //for (int i = 0; i < top_temp_.count(); ++i)
    //  LOG(INFO)<<top_data_temp[i]<<"\n";
    //LOG(INFO)<<"stage4\n";
  }

  // final(first) dimension to tile
  s = 0;
  Dtype* input_data = top_temp_.mutable_cpu_data();
  Dtype* output_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < outer_dim_[s]; ++i) {
    for (int t = 0; t < tiles_[s]; ++t) {
      caffe_copy(top_inner_dim_[s], input_data, output_data);
      output_data += top_inner_dim_[s];
    }
    input_data += top_inner_dim_[s];
  }
}

template <typename Dtype>
void TileNDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for(int s=outer_dim_.size()-1;s>=0;s--)
  {
    for (int i = 0; i < outer_dim_[s]; ++i) {
      caffe_copy(inner_dim_[s], top_diff, bottom_diff);
      top_diff += inner_dim_[s];
      for (int t = 1; t < tiles_[s]; ++t) {
        caffe_axpy(inner_dim_[s], Dtype(1), top_diff, bottom_diff);
        top_diff += inner_dim_[s];
      }
      bottom_diff += inner_dim_[s];
    }
  }
}

//#ifdef CPU_ONLY
//STUB_GPU(TileLayer);
//#endif

INSTANTIATE_CLASS(TileNDLayer);
REGISTER_LAYER_CLASS(TileND);

}  // namespace caffe
