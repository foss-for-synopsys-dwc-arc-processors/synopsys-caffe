#ifndef CAFFE_SHUFFLE_CHANNEL_LAYER_HPP_
#define CAFFE_SHUFFLE_CHANNEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Merged from https://github.com/farmingyard/ShuffleNet
 * This layer is used for reorder the data at channel layer.
 * Could replace the 3 layers' functionality in ShuffleNet. Example:
 * use ShuffleChannel layer group=2, equals to
 * [1 232 14 14] Reshape(5D)-> [1 2 116 14 14] Permute(5D)-> [1 116 2 14 14] Reshape(4D)-> [1 232 14 14]
 */

template <typename Dtype>
class ShuffleChannelLayer : public Layer<Dtype> {
public:
  explicit ShuffleChannelLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "ShuffleChannel"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  void Resize_cpu(Dtype *output, const Dtype *input, int group_row, int group_column, int len);
  void Resize_gpu(Dtype *output, const Dtype *input, int group_row, int group_column, int len);

  //Blob<Dtype> temp_blob_;
  int group_;
};

}  // namespace caffe

#endif  // CAFFE_SHUFFLE_CHANNEL_LAYER_HPP_
