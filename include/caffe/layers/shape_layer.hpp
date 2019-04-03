#ifndef CAFFE_SHAPE_LAYER_HPP_
#define CAFFE_SHAPE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ShapeLayer : public Layer<Dtype> {
public:
	explicit ShapeLayer(const LayerParameter& param) :
			Layer<Dtype>(param) {
	}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {return "Shape";}
	virtual inline int ExactBottomBlobs() const {return 1;}
	virtual inline int ExactNumTopBlobs() const {return 1;}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}

};

}  // namespace caffe

#endif  // CAFFE_SHAPE_LAYER_HPP_
