#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/peephole_lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PeepholeLSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void PeepholeLSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
   names->resize(2);
  (*names)[0] = "h_" + format_int(this->T_);
  (*names)[1] = "c_T";
}

template <typename Dtype>
void PeepholeLSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
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
void PeepholeLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h";
}

template <typename Dtype>
void PeepholeLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
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

  LayerParameter sigmoid_param;
  sigmoid_param.set_type("Sigmoid");

  LayerParameter tanh_param;
  tanh_param.set_type("TanH");

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

  LayerParameter concat_layer;
  concat_layer.set_type("Concat");
  concat_layer.mutable_concat_param()->set_axis(0);


  LayerParameter output_concat_layer;
  output_concat_layer.CopyFrom(concat_layer);
  output_concat_layer.set_name("h_concat");
  output_concat_layer.add_top("h");

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
    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     c_conted_{t-1} := cont_t * c_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
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

    // Add layer to compute
    //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param()->set_name("W_hc");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hc_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     non_c_sum_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* inner_non_c_sum_layer = net_param->add_layer();
      inner_non_c_sum_layer->CopyFrom(sum_param);
      inner_non_c_sum_layer->set_name("inner_non_c_sum_" + ts);
      inner_non_c_sum_layer->add_bottom("W_hc_h_" + tm1s);
      inner_non_c_sum_layer->add_bottom("W_xc_x_" + ts);
      if (this->static_input_) {
        inner_non_c_sum_layer->add_bottom("W_xc_x_static");
      }
      inner_non_c_sum_layer->add_top("inner_non_c_sum_" + ts);
    }

    {
      LayerParameter* inner_non_c_slice_param = net_param->add_layer();
      inner_non_c_slice_param->CopyFrom(slice_param);
      inner_non_c_slice_param->set_name("inner_non_c_slice" + ts);
      inner_non_c_slice_param->add_bottom("inner_non_c_sum_" + ts);
      inner_non_c_slice_param->add_top("W_xc_x_i_" + ts);
      inner_non_c_slice_param->add_top("W_xc_x_o_" + ts);
      inner_non_c_slice_param->add_top("W_xc_x_f_" + ts);
      inner_non_c_slice_param->add_top("W_xc_x_c_" + ts);
      inner_non_c_slice_param->mutable_slice_param()->set_axis(2);
    }
    // Pi (.) Ct-1
    {
      LayerParameter* Pi_c_tm1s_param = net_param->add_layer();
      Pi_c_tm1s_param->CopyFrom(scale_param);
      Pi_c_tm1s_param->set_name("Pi_c_tm1s_" + ts);
      Pi_c_tm1s_param->add_param()->set_name("P_i");
      Pi_c_tm1s_param->add_bottom("c_conted_" + tm1s);
      Pi_c_tm1s_param->add_top("Pi_c_tm1s_" + ts);
      Pi_c_tm1s_param->mutable_scale_param()->set_axis(-1);
    }
    // - it = Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi
    {
      LayerParameter* inner_i_param = net_param->add_layer();
      inner_i_param->CopyFrom(sum_param);
      inner_i_param->set_name("inner_i_" + ts);
      inner_i_param->add_top("inner_i_" + ts);
      inner_i_param->add_bottom("W_xc_x_i_" + ts);
      inner_i_param->add_bottom("Pi_c_tm1s_" + ts);
    }
    // - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    {
      LayerParameter* f_i_param = net_param->add_layer();
      f_i_param->CopyFrom(sigmoid_param);
      f_i_param->add_bottom("inner_i_" + ts);
      f_i_param->add_top("i_" + ts);
      f_i_param->set_name("i_" + ts);
    }
    // Pf (.) Ct-1
    {
      LayerParameter* Pf_c_tm1s_param = net_param->add_layer();
      Pf_c_tm1s_param->CopyFrom(scale_param);
      Pf_c_tm1s_param->set_name("Pf_c_tm1s_" + ts);
      Pf_c_tm1s_param->add_param()->set_name("P_f");
      Pf_c_tm1s_param->add_bottom("c_conted_" + tm1s);
      Pf_c_tm1s_param->add_top("Pf_c_tm1s_" + ts);
      Pf_c_tm1s_param->mutable_scale_param()->set_axis(-1);
    }
    // - ft = Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf
    {
      LayerParameter* inner_f_param = net_param->add_layer();
      inner_f_param->CopyFrom(sum_param);
      inner_f_param->set_name("inner_f_" + ts);
      inner_f_param->add_top("inner_f_" + ts);
      inner_f_param->add_bottom("W_xc_x_f_" + ts);
      inner_f_param->add_bottom("Pf_c_tm1s_" + ts);
    }
    // - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    {
      LayerParameter* f_f_param = net_param->add_layer();
      f_f_param->CopyFrom(sigmoid_param);
      f_f_param->add_bottom("inner_f_" + ts);
      f_f_param->add_top("f_" + ts);
      f_f_param->set_name("f_" + ts);
    }
    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    {
      LayerParameter* c_half_param = net_param->add_layer();
      c_half_param->CopyFrom(tanh_param);
      c_half_param->set_name("c_half_" + ts);
      c_half_param->add_bottom("W_xc_x_c_" + ts);
      c_half_param->add_top("c_half_" + ts);
    }
    // it (.) ct
    {
      LayerParameter* it_ct_prod_parm = net_param->add_layer();
      it_ct_prod_parm->CopyFrom(prod_param);
      it_ct_prod_parm->set_name("it_ct_prod_" + ts);
      it_ct_prod_parm->add_bottom("i_" + ts);
      it_ct_prod_parm->add_bottom("c_half_" + ts);
      it_ct_prod_parm->add_top("it_ct_prod_" + ts);
    }
    // ft (.) Ct-1
    {
      LayerParameter* ft_c_tm1s_param = net_param->add_layer();
      ft_c_tm1s_param->CopyFrom(prod_param);
      ft_c_tm1s_param->set_name("ft_c_tm1s_prod_" + ts);
      ft_c_tm1s_param->add_bottom("f_" + ts);
      ft_c_tm1s_param->add_bottom("c_conted_" + tm1s);
      ft_c_tm1s_param->add_top("ft_c_tm1s_prod_" + ts);
    }
    // Ct = ft (.) Ct-1 + it (.) ct
    {
      LayerParameter* c_t_param = net_param->add_layer();
      c_t_param->CopyFrom(sum_param);
      c_t_param->set_name("c_" + ts);
      c_t_param->add_top("c_" + ts);
      c_t_param->add_bottom("it_ct_prod_" + ts);
      c_t_param->add_bottom("ft_c_tm1s_prod_" + ts);

    }
    // Po (.) Ct
    {
      LayerParameter* Po_ct_param = net_param->add_layer();
      Po_ct_param->CopyFrom(scale_param);
      Po_ct_param->set_name("Po_ct_" + ts);
      Po_ct_param->add_param()->set_name("P_o");
      Po_ct_param->add_bottom("c_" + ts);
      Po_ct_param->add_top("Po_ct_" + ts);
      Po_ct_param->mutable_scale_param()->set_axis(-1);
    }
    // Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo
    {
      LayerParameter* inner_o_param = net_param->add_layer();
      inner_o_param->CopyFrom(sum_param);
      inner_o_param->set_name("inner_o_" + ts);
      inner_o_param->add_top("inner_o_" + ts);
      inner_o_param->add_bottom("Po_ct_" + ts);
      inner_o_param->add_bottom("W_xc_x_o_" + ts);

    }
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    {
      LayerParameter* f_o_param = net_param->add_layer();
      f_o_param->CopyFrom(sigmoid_param);
      f_o_param->add_bottom("inner_o_" + ts);
      f_o_param->add_top("o_" + ts);
      f_o_param->set_name("o_" + ts);
    }
    // h(Ct)
    {
      LayerParameter* h_ct_param = net_param->add_layer();
      h_ct_param->CopyFrom(tanh_param);
      h_ct_param->add_top("h_ct_" + ts);
      h_ct_param->add_bottom("c_" + ts);
      h_ct_param->set_name("h_ct_" + ts);
    }
    // Ht = ot (.) h(Ct)
    {
      LayerParameter* h_layer_param = net_param->add_layer();
      h_layer_param->CopyFrom(prod_param);
      h_layer_param->set_name("h_" + ts);
      h_layer_param->add_bottom("o_" + ts);
      h_layer_param->add_bottom("h_ct_" + ts);
      h_layer_param->add_top("h_" + ts);
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

INSTANTIATE_CLASS(PeepholeLSTMLayer);
REGISTER_LAYER_CLASS(PeepholeLSTM);

}  // namespace caffe
