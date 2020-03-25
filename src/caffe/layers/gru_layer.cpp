#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/gru_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GRULayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h_0";
}

template <typename Dtype>
void GRULayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h_" + format_int(this->T_);
}

template <typename Dtype>
void GRULayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  shapes->resize(1);
  (*shapes)[0].Clear();
  (*shapes)[0].add_dim(1);  // a single timestep
  (*shapes)[0].add_dim(this->N_);
  (*shapes)[0].add_dim(num_output);
}

template <typename Dtype>
void GRULayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h";
}

template <typename Dtype>
void GRULayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  const int linear_before_reset = this->layer_param_.recurrent_param().linear_before_reset();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output * 3);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(2);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter prod_param;
  prod_param.set_type("Eltwise");
  prod_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_PROD);

  LayerParameter scale_param;
  scale_param.set_type("Scale");
  scale_param.mutable_scale_param()->set_axis(0);

  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  LayerParameter F_activation_param;
  if (this->activations_.size() > 0)
    F_activation_param.set_type(this->activations_[0]);
  else
    F_activation_param.set_type("Sigmoid");

  LayerParameter G_activation_param;
  if (this->activations_.size() > 1)
    G_activation_param.set_type(this->activations_[1]);
  else
    G_activation_param.set_type("TanH");

  LayerParameter split_param;
  split_param.set_type("Split");

  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(1, input_shapes.size());

  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");

  InputParameter* input_param = input_layer_param->mutable_input_param();
  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(0);

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xc");
    x_transform_param->add_param()->set_name("b_c");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xc_x");
    x_transform_param->add_propagate_down(true);
  }

  if (this->static_input_) {
    // Add layer to transform x_static to the gate dimension.
    //     W_xc_x_static = W_xc_static * x_static
    LayerParameter* x_static_transform_param = net_param->add_layer();
    x_static_transform_param->CopyFrom(hidden_param);
    x_static_transform_param->mutable_inner_product_param()->set_axis(1);
    x_static_transform_param->set_name("W_xc_x_static");
    x_static_transform_param->add_param()->set_name("W_xc_static");
    x_static_transform_param->add_bottom("x_static");
    x_static_transform_param->add_top("W_xc_x_static_preshape");
    x_static_transform_param->add_propagate_down(true);

    LayerParameter* reshape_param = net_param->add_layer();
    reshape_param->set_type("Reshape");
    BlobShape* new_shape =
         reshape_param->mutable_reshape_param()->mutable_shape();
    new_shape->add_dim(1);  // One timestep.
    // Should infer this->N as the dimension so we can reshape on batch size.
    new_shape->add_dim(-1);
    new_shape->add_dim(
        x_static_transform_param->inner_product_param().num_output());
    reshape_param->set_name("W_xc_x_static_reshape");
    reshape_param->add_bottom("W_xc_x_static_preshape");
    reshape_param->add_top("W_xc_x_static");
  }

  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("W_xc_x");
  x_slice_param->set_name("W_xc_x_slice");

  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("h");
  output_concat_layer.mutable_concat_param()->set_axis(0);

  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = format_int(t - 1);
    string ts = format_int(t);

    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("W_xc_x_" + ts);
    if (this->static_input_) {
      LayerParameter* X_static_param = net_param->add_layer();
      X_static_param->CopyFrom(sum_param);
      X_static_param->set_name("XW_X_static_" + ts);
      X_static_param->add_bottom("W_xc_x_" + ts);
      X_static_param->add_bottom("W_xc_x_static");
      X_static_param->add_top("XW_X_static_" + ts);
    }

    // Slice X * W into three parts:
    // 1. R
    // 2. Z
    // 3. H
    {
      LayerParameter* wx_slice_param = net_param->add_layer();
      wx_slice_param->CopyFrom(slice_param);
      wx_slice_param->set_name("wx_slice" + ts);
      if (this->static_input_)
        wx_slice_param->add_bottom("XW_X_static_" + ts);
      else
        wx_slice_param->add_bottom("W_xc_x_" + ts);
      wx_slice_param->add_top("W_xc_x_z_" + ts);
      wx_slice_param->add_top("W_xc_x_r_" + ts);
      wx_slice_param->add_top("W_xc_x_h_" + ts);
      wx_slice_param->mutable_slice_param()->set_axis(2);
    }

    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(scale_param);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s);
    }

    // Add layer to compute
    //     R_z_h_{t-1} := R_z * h_conted_{t-1}
    {
      LayerParameter* R_z_h_param = net_param->add_layer();
      R_z_h_param->CopyFrom(hidden_param);
      R_z_h_param->set_name("R_z_h_" + ts);
      R_z_h_param->add_param()->set_name("R_z");
      R_z_h_param->add_bottom("h_conted_" + tm1s);
      R_z_h_param->add_top("R_z_h_" + ts);
      R_z_h_param->mutable_inner_product_param()->set_num_output(num_output);
      R_z_h_param->mutable_inner_product_param()->set_axis(2);
    }
    // Compute the inner part of Z:
    //      R_z * h_conted_{t-1} + W_xc * x_t + b_c
    {
      LayerParameter* inner_z_sum_param = net_param->add_layer();
      inner_z_sum_param->CopyFrom(sum_param);
      inner_z_sum_param->set_name("inner_z_" + ts);
      inner_z_sum_param->add_bottom("R_z_h_" + ts);
      inner_z_sum_param->add_bottom("W_xc_x_z_" + ts);
      inner_z_sum_param->add_top("inner_z_" + ts);
    }
    // - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    {
      LayerParameter* f_z_param = net_param->add_layer();
      f_z_param->CopyFrom(F_activation_param);
      f_z_param->add_bottom("inner_z_" + ts);
      f_z_param->add_top("z_" + ts);
      f_z_param->set_name("z_" + ts);
    }
    // Add layer to compute
    //     R_r_h_{t-1} := R_r * h_conted_{t-1}
    {
      LayerParameter* R_r_h_param = net_param->add_layer();
      R_r_h_param->CopyFrom(hidden_param);
      R_r_h_param->set_name("R_r_h_" + ts);
      R_r_h_param->add_param()->set_name("R_r");
      R_r_h_param->add_bottom("h_conted_" + tm1s);
      R_r_h_param->add_top("R_r_h_" + ts);
      R_r_h_param->mutable_inner_product_param()->set_num_output(num_output);
      R_r_h_param->mutable_inner_product_param()->set_axis(2);
    }
    // Compute the inner part of R:
    //      R_r * h_conted_{t-1} + W_xc * x_t + b_c
    {
      LayerParameter* inner_r_sum_param = net_param->add_layer();
      inner_r_sum_param->CopyFrom(sum_param);
      inner_r_sum_param->set_name("inner_r_" + ts);
      inner_r_sum_param->add_bottom("R_r_h_" + ts);
      inner_r_sum_param->add_bottom("W_xc_x_r_" + ts);
      inner_r_sum_param->add_top("inner_r_" + ts);
    }
    // - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    {
      LayerParameter* f_r_param = net_param->add_layer();
      f_r_param->CopyFrom(F_activation_param);
      f_r_param->add_bottom("inner_r_" + ts);
      f_r_param->add_top("r_" + ts);
      f_r_param->set_name("r_" + ts);
    }

    // (rt (.) Ht-1)*(Rh^T)
    // # default, when linear_before_reset = 0
    if (linear_before_reset == 0){
      LayerParameter* r_ts_h_tm1s_layer = net_param->add_layer();
      r_ts_h_tm1s_layer->CopyFrom(prod_param);
      r_ts_h_tm1s_layer->set_name("r_ts_h_tm1s_" + ts);
      r_ts_h_tm1s_layer->add_bottom("r_" + ts);
      r_ts_h_tm1s_layer->add_bottom("h_conted_" + tm1s);
      r_ts_h_tm1s_layer->add_top("r_ts_h_tm1s_" + ts);

      LayerParameter* r_ts_h_tm1s_R_layer = net_param->add_layer();
      r_ts_h_tm1s_R_layer->CopyFrom(hidden_param);
      r_ts_h_tm1s_R_layer->mutable_inner_product_param()->set_num_output(num_output);
      r_ts_h_tm1s_R_layer->set_name("R_rt_h_tm1s_" + ts);
      r_ts_h_tm1s_R_layer->add_param()->set_name("R_h");
      r_ts_h_tm1s_R_layer->add_bottom("r_ts_h_tm1s_" + ts);
      r_ts_h_tm1s_R_layer->add_top("R_rt_h_tm1s_" + ts);
    } else {
      // (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh
      // # when linear_before_reset != 0
      LayerParameter* r_ts_h_tm1s_R_layer = net_param->add_layer();
      r_ts_h_tm1s_R_layer->CopyFrom(biased_hidden_param);
      r_ts_h_tm1s_R_layer->mutable_inner_product_param()->set_num_output(num_output);
      r_ts_h_tm1s_R_layer->set_name("Rh_h_tm1s_" + ts);
      r_ts_h_tm1s_R_layer->add_bottom("h_conted_" + tm1s);
      r_ts_h_tm1s_R_layer->add_top("Rh_h_tm1s_" + ts);
      r_ts_h_tm1s_R_layer->add_param()->set_name("R_h");
      r_ts_h_tm1s_R_layer->add_param()->set_name("Rb_h");

      LayerParameter* r_ts_h_tm1s_layer = net_param->add_layer();
      r_ts_h_tm1s_layer->CopyFrom(prod_param);
      r_ts_h_tm1s_layer->set_name("R_rt_h_tm1s_" + ts);
      r_ts_h_tm1s_layer->add_bottom("r_" + ts);
      r_ts_h_tm1s_layer->add_bottom("Rh_h_tm1s_" + ts);
      r_ts_h_tm1s_layer->add_top("R_rt_h_tm1s_" + ts);
    }
    // - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
    // # default, when linear_before_reset = 0
    // - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
    // # when linear_before_reset != 0
    {
      LayerParameter* h_sum_layer = net_param->add_layer();
      h_sum_layer->CopyFrom(sum_param);
      h_sum_layer->set_name("inner_h_" + ts);
      h_sum_layer->add_bottom("R_rt_h_tm1s_" + ts);
      h_sum_layer->add_bottom("W_xc_x_h_" + ts);
      h_sum_layer->add_top("inner_h_" + ts);
    }
    {
      LayerParameter* g_layer = net_param->add_layer();
      g_layer->CopyFrom(G_activation_param);
      g_layer->add_bottom("inner_h_" + ts);
      g_layer->add_top("ht_" + ts);
      g_layer->set_name("ht_" + ts);
    }
    // - Ht = ht + zt (.) (Ht-1 - ht)
    {
      LayerParameter* H_h_sub_layer = net_param->add_layer();
      H_h_sub_layer->CopyFrom(sum_param);
      H_h_sub_layer->set_name("H_h_sub_" + ts);
      H_h_sub_layer->add_bottom("h_conted_" + tm1s);
      H_h_sub_layer->add_bottom("ht_" + ts);
      H_h_sub_layer->mutable_eltwise_param()->add_coeff(1);
      H_h_sub_layer->mutable_eltwise_param()->add_coeff(-1);
      H_h_sub_layer->add_top("H_h_sub_" + ts);
    }
    {
      LayerParameter* z_h_prod_layer = net_param->add_layer();
      z_h_prod_layer->CopyFrom(prod_param);
      z_h_prod_layer->set_name("z_h_prod_" + ts);
      z_h_prod_layer->add_bottom("z_" + ts);
      z_h_prod_layer->add_bottom("H_h_sub_" + ts);
      z_h_prod_layer->add_top("z_h_prod_" + ts);
    }
    {
      LayerParameter* h_layer = net_param->add_layer();
      h_layer->CopyFrom(sum_param);
      h_layer->set_name("h_" + ts);
      h_layer->add_bottom("ht_" + ts);
      h_layer->add_bottom("z_h_prod_" + ts);
      h_layer->add_top("h_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
  }  // for (int t = 1; t <= this->T_; ++t)

  net_param->add_layer()->CopyFrom(output_concat_layer);
}

INSTANTIATE_CLASS(GRULayer);
REGISTER_LAYER_CLASS(GRU);

}  // namespace caffe
