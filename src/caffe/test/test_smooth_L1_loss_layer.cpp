#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/smooth_L1_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SmoothL1LossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SmoothL1LossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_inside_weights_(new Blob<Dtype>(10, 5, 1, 1)), //Faster RCNN
        blob_bottom_outside_weights_(new Blob<Dtype>(10, 5, 1, 1)), //Faster RCNN
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
	//FillerParameter const_filler_param;
	//const_filler_param.set_value(-1.);
	//ConstantFiller<Dtype> const_filler(const_filler_param);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_SSD_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_bottom_vec_SSD_.push_back(blob_bottom_label_);
    //const_filler.Fill(this->blob_bottom_inside_weights_); //Faster RCNN
    filler.Fill(this->blob_bottom_inside_weights_);
    blob_bottom_vec_.push_back(blob_bottom_inside_weights_);
    //const_filler.Fill(this->blob_bottom_outside_weights_); //Faster RCNN
    filler.Fill(this->blob_bottom_outside_weights_);
    blob_bottom_vec_.push_back(blob_bottom_outside_weights_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_SSD_.push_back(blob_top_loss_);
  }
  virtual ~SmoothL1LossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_inside_weights_; //Faster RCNN
    delete blob_bottom_outside_weights_; //Faster RCNN
    delete blob_top_loss_;
  }

  void TestForward(bool SSD) {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    if (!SSD) { //Faster RCNN
      SmoothL1LossParameter* loss_param = layer_param.mutable_smooth_l1_loss_param();
      loss_param->set_sigma(2.4); //Faster RCNN
      loss_param->set_abssum(false); //Faster RCNN
    }
    SmoothL1LossLayer<Dtype> layer_weight_1(layer_param);
    Dtype loss_weight_1;
    if (SSD) {
      layer_weight_1.SetUp(this->blob_bottom_vec_SSD_, this->blob_top_vec_SSD_);
      loss_weight_1 = layer_weight_1.Forward(this->blob_bottom_vec_SSD_, this->blob_top_vec_SSD_);
    }
    else {
      layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      loss_weight_1 = layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    }
    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    SmoothL1LossLayer<Dtype> layer_weight_2(layer_param);
    Dtype loss_weight_2;
    if (SSD) {
      layer_weight_2.SetUp(this->blob_bottom_vec_SSD_, this->blob_top_vec_SSD_);
      loss_weight_2 = layer_weight_2.Forward(this->blob_bottom_vec_SSD_, this->blob_top_vec_SSD_);
    }
    else {
      layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      loss_weight_2 = layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    }
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_inside_weights_; //Faster RCNN
  Blob<Dtype>* const blob_bottom_outside_weights_; //Faster RCNN
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_SSD_; //SSD
  vector<Blob<Dtype>*> blob_top_vec_SSD_; //SSD
};

TYPED_TEST_CASE(SmoothL1LossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SmoothL1LossLayerTest, TestForward_SSD) {
  this->TestForward(true);
}

TYPED_TEST(SmoothL1LossLayerTest, TestForward_FCN) { //Faster RCNN
  this->TestForward(false);
}

TYPED_TEST(SmoothL1LossLayerTest, TestGradient_SSD) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SmoothL1LossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_SSD_, this->blob_top_vec_SSD_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_SSD_,
      this->blob_top_vec_SSD_);
}

TYPED_TEST(SmoothL1LossLayerTest, TestGradient_FCN) { //Faster RCNN
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmoothL1LossParameter* loss_param = //Faster RCNN
      layer_param.mutable_smooth_l1_loss_param();
  loss_param->set_sigma(2.4); //Faster RCNN
  loss_param->set_abssum(false); //Faster RCNN
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SmoothL1LossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0); //Faster RCNN
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1); //Faster RCNN
}

}  // namespace caffe
