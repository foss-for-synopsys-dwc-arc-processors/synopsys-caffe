#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/lstm_v2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LSTMV2Layer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void LSTMV2Layer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_" + format_int(this->T_);
  (*names)[1] = "c_T";
}

template <typename Dtype>
void LSTMV2Layer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  const int num_blobs = 2;
  shapes->resize(num_blobs);
  for (int i = 0; i < num_blobs; ++i) {
    (*shapes)[i].Clear();
    (*shapes)[i].add_dim(1);  // a single timestep
    (*shapes)[i].add_dim(this->N_);
    (*shapes)[i].add_dim(num_output);
  }
}

template <typename Dtype>
void LSTMV2Layer<Dtype>::
OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h";
}

template <typename Dtype>
void LSTMV2Layer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
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

  LayerParameter H_activation_param;
  if (this->activations_.size() > 2)
    H_activation_param.set_type(this->activations_[2]);
  else
    H_activation_param.set_type("TanH");

  LayerParameter prod_param;
  prod_param.set_type("Eltwise");
  prod_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_PROD);

  LayerParameter split_param;
  split_param.set_type("Split");

  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(2, input_shapes.size());

  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();

  input_layer_param->add_top("c_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[1]);

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
    //     R_h_{t-1} := R * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param()->set_name("R");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("R_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := R * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = R_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name("gate_input_" + ts);
      input_sum_layer->add_bottom("R_h_" + tm1s);
      input_sum_layer->add_bottom("W_xc_x_" + ts);
      if (this->static_input_) {
        input_sum_layer->add_bottom("W_xc_x_static");
      }
      input_sum_layer->add_top("gate_input_" + ts);
    }
    //     [ i_t' ]
    //     [ o_t' ] := gate_input_t
    //     [ f_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         o_t := \sigmoid[o_t']
    //         f_t := \sigmoid[f_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* inner_non_c_slice_param = net_param->add_layer();
      inner_non_c_slice_param->CopyFrom(slice_param);
      inner_non_c_slice_param->set_name("gate_input_slice_" + ts);
      inner_non_c_slice_param->add_bottom("gate_input_" + ts);
      inner_non_c_slice_param->add_top("gate_input_i_" + ts);
      inner_non_c_slice_param->add_top("gate_input_o_" + ts);
      inner_non_c_slice_param->add_top("gate_input_f_" + ts);
      inner_non_c_slice_param->add_top("gate_input_g_" + ts);
      inner_non_c_slice_param->mutable_slice_param()->set_axis(2);
    }
    {
      LayerParameter* i_t_param = net_param->add_layer();
      i_t_param->CopyFrom(F_activation_param);
      i_t_param->add_bottom("gate_input_i_" + ts);
      i_t_param->add_top("i_" + ts);
      i_t_param->set_name("i_" + ts);
    }
    {
      LayerParameter* o_t_param = net_param->add_layer();
      o_t_param->CopyFrom(F_activation_param);
      o_t_param->add_bottom("gate_input_o_" + ts);
      o_t_param->add_top("o_" + ts);
      o_t_param->set_name("o_" + ts);
    }
    {
      LayerParameter* f_t_param = net_param->add_layer();
      f_t_param->CopyFrom(F_activation_param);
      f_t_param->add_bottom("gate_input_f_" + ts);
      f_t_param->add_top("f_" + ts);
      f_t_param->set_name("f_" + ts);
    }
    {
      LayerParameter * g_t_param = net_param->add_layer();
      g_t_param->CopyFrom(G_activation_param);
      g_t_param->add_bottom("gate_input_g_" + ts);
      g_t_param->add_top("g_" + ts);
      g_t_param->set_name("g_" + ts);
    }
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     c_conted_{t-1} := cont_t * c_{t-1}
    //     c_conted_{t-1} := c_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_c_param = net_param->add_layer();
      cont_c_param->CopyFrom(scale_param);
      cont_c_param->set_name("c_conted_" + tm1s);
      cont_c_param->add_bottom("c_" + tm1s);
      cont_c_param->add_bottom("cont_" + ts);
      cont_c_param->add_top("c_conted_" + tm1s);
    }
    // f_t (.) c_{t-1}
    {
      LayerParameter* f_c_tm1s_prod_parm = net_param->add_layer();
      f_c_tm1s_prod_parm->CopyFrom(prod_param);
      f_c_tm1s_prod_parm->set_name("f_c_tm1s_prod_" + ts);
      f_c_tm1s_prod_parm->add_bottom("f_" + ts);
      f_c_tm1s_prod_parm->add_bottom("c_conted_" + tm1s);
      f_c_tm1s_prod_parm->add_top("f_c_tm1s_prod_" + ts);
    }
    // i_t (.) g_t
    {
      LayerParameter* i_g_prod_parm = net_param->add_layer();
      i_g_prod_parm->CopyFrom(prod_param);
      i_g_prod_parm->set_name("i_g_prod_" + ts);
      i_g_prod_parm->add_bottom("i_" + ts);
      i_g_prod_parm->add_bottom("g_" + ts);
      i_g_prod_parm->add_top("i_g_prod_" + ts);
    }
    {
      LayerParameter* c_sum_layer = net_param->add_layer();
      c_sum_layer->CopyFrom(sum_param);
      c_sum_layer->set_name("c_" + ts);
      c_sum_layer->add_bottom("f_c_tm1s_prod_" + ts);
      c_sum_layer->add_bottom("i_g_prod_" + ts);
      c_sum_layer->add_top("c_" + ts);
    }
    {
      LayerParameter * H_c_param = net_param->add_layer();
      H_c_param->CopyFrom(H_activation_param);
      H_c_param->add_bottom("c_" + ts);
      H_c_param->add_top("H_c_" + ts);
      H_c_param->set_name("H_c_" + ts);
    }
    {
      LayerParameter* h_parm = net_param->add_layer();
      h_parm->CopyFrom(prod_param);
      h_parm->set_name("h_" + ts);
      h_parm->add_bottom("o_" + ts);
      h_parm->add_bottom("H_c_" + ts);
      h_parm->add_top("h_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
  }  // for (int t = 1; t <= this->T_; ++t)

  {
    LayerParameter* c_T_copy_param = net_param->add_layer();
    c_T_copy_param->CopyFrom(split_param);
    c_T_copy_param->add_bottom("c_" + format_int(this->T_));
    c_T_copy_param->add_top("c_T");
  }
  net_param->add_layer()->CopyFrom(output_concat_layer);
}

INSTANTIATE_CLASS(LSTMV2Layer);
REGISTER_LAYER_CLASS(LSTMV2);

}  // namespace caffe
