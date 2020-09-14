#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rnn_v2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RNNv2Layer<Dtype>::RecurrentInputBlobNames(vector<string> *names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void RNNv2Layer<Dtype>::RecurrentOutputBlobNames(vector<string> *names) const {
  names->resize(2);
  (*names)[0] = "h_" + format_int(this->T_);
  (*names)[1] = "c_T";
}

template <typename Dtype>
void RNNv2Layer<Dtype>::RecurrentInputShapes(vector<BlobShape> *shapes) const {
  const int hidden_size = this->layer_param_.rnn_v2_param().hidden_size();
  const int num_blobs = 2;
  string direction = this->layer_param_.rnn_v2_param().direction();
  shapes->resize(num_blobs);
  for (int i = 0; i < num_blobs; ++i) {
    (*shapes)[i].Clear();
    if(direction == "bidirectional")
      (*shapes)[i].add_dim(2);
    else
      (*shapes)[i].add_dim(1);
    (*shapes)[i].add_dim(this->N_);
    (*shapes)[i].add_dim(hidden_size);
  }
}

template <typename Dtype>
void RNNv2Layer<Dtype>::OutputBlobNames(vector<string> *names) const {
  names->resize(1);
  (*names)[0] = "h";
}

template <typename Dtype>
void RNNv2Layer<Dtype>::FillUnrolledNet(
    NetParameter *net_param,
    string x_name,
    string cont_name,
    vector<string> recur_input_names,
    vector<string> output_names,
    vector<string> recur_output_names,
    const string &name_prefix) {
  const int hidden_size = this->layer_param_.rnn_v2_param().hidden_size();
  CHECK_GT(hidden_size, 0) << "hidden_size must be positive";

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(hidden_size * 4);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(2);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);

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

  LayerParameter *input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  InputParameter *input_param = input_layer_param->mutable_input_param();

  input_layer_param->add_top(name_prefix + recur_input_names[0]);
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  input_layer_param->add_top(name_prefix + recur_input_names[1]);
  input_param->add_shape()->CopyFrom(input_shapes[1]);

  LayerParameter *cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name(name_prefix + "cont_slice");
  cont_slice_param->add_bottom(name_prefix + cont_name);
  cont_slice_param->mutable_slice_param()->set_axis(0);

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
  {
    LayerParameter *x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name(name_prefix + "x_transform");
    x_transform_param->add_param()->set_name(name_prefix + "W_xc");
    x_transform_param->add_param()->set_name(name_prefix + "b_c");
    x_transform_param->add_bottom(name_prefix + x_name);
    x_transform_param->add_top(name_prefix + "W_xc_x");
    x_transform_param->add_propagate_down(true);
  }

  LayerParameter *x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom(name_prefix + "W_xc_x");
  x_slice_param->set_name(name_prefix + "W_xc_x_slice");

  LayerParameter output_concat_layer;
  output_concat_layer.set_name(name_prefix + "h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top(name_prefix + output_names[0]);
  output_concat_layer.mutable_concat_param()->set_axis(0);

  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = format_int(t - 1);
    string ts = format_int(t);

    cont_slice_param->add_top(name_prefix + "cont_" + ts);
    x_slice_param->add_top(name_prefix + "W_xc_x_" + ts);

    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter *cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(scale_param);
      cont_h_param->set_name(name_prefix + "h_conted_" + tm1s);
      cont_h_param->add_bottom(name_prefix + "h_" + tm1s);
      cont_h_param->add_bottom(name_prefix + "cont_" + ts);
      cont_h_param->add_top(name_prefix + "h_conted_" + tm1s);
    }

    // Add layer to compute
    //     R_h_{t-1} := R * h_conted_{t-1}
    {
      LayerParameter *w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name(name_prefix + "transform_" + ts);
      w_param->add_param()->set_name(name_prefix + "R");
      w_param->add_bottom(name_prefix + "h_conted_" + tm1s);
      w_param->add_top(name_prefix + "R_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := R * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = R_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter *input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name(name_prefix + "gate_input_" + ts);
      input_sum_layer->add_bottom(name_prefix + "R_h_" + tm1s);
      input_sum_layer->add_bottom(name_prefix + "W_xc_x_" + ts);
      input_sum_layer->add_top(name_prefix + "gate_input_" + ts);
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
      LayerParameter *inner_non_c_slice_param = net_param->add_layer();
      inner_non_c_slice_param->CopyFrom(slice_param);
      inner_non_c_slice_param->set_name(name_prefix + "gate_input_slice_" + ts);
      inner_non_c_slice_param->add_bottom(name_prefix + "gate_input_" + ts);
      inner_non_c_slice_param->add_top(name_prefix + "gate_input_i_" + ts);
      inner_non_c_slice_param->add_top(name_prefix + "gate_input_o_" + ts);
      inner_non_c_slice_param->add_top(name_prefix + "gate_input_f_" + ts);
      inner_non_c_slice_param->add_top(name_prefix + "gate_input_g_" + ts);
      inner_non_c_slice_param->mutable_slice_param()->set_axis(2);
    }
    {
      LayerParameter *i_t_param = net_param->add_layer();
      i_t_param->CopyFrom(F_activation_param);
      i_t_param->add_bottom(name_prefix + "gate_input_i_" + ts);
      i_t_param->add_top(name_prefix + "i_" + ts);
      i_t_param->set_name(name_prefix + "i_" + ts);
    }
    {
      LayerParameter *o_t_param = net_param->add_layer();
      o_t_param->CopyFrom(F_activation_param);
      o_t_param->add_bottom(name_prefix + "gate_input_o_" + ts);
      o_t_param->add_top(name_prefix + "o_" + ts);
      o_t_param->set_name(name_prefix + "o_" + ts);
    }
    {
      LayerParameter *f_t_param = net_param->add_layer();
      f_t_param->CopyFrom(F_activation_param);
      f_t_param->add_bottom(name_prefix + "gate_input_f_" + ts);
      f_t_param->add_top(name_prefix + "f_" + ts);
      f_t_param->set_name(name_prefix + "f_" + ts);
    }
    {
      LayerParameter *g_t_param = net_param->add_layer();
      g_t_param->CopyFrom(G_activation_param);
      g_t_param->add_bottom(name_prefix + "gate_input_g_" + ts);
      g_t_param->add_top(name_prefix + "g_" + ts);
      g_t_param->set_name(name_prefix + "g_" + ts);
    }
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     c_conted_{t-1} := cont_t * c_{t-1}
    //     c_conted_{t-1} := c_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter *cont_c_param = net_param->add_layer();
      cont_c_param->CopyFrom(scale_param);
      cont_c_param->set_name(name_prefix + "c_conted_" + tm1s);
      cont_c_param->add_bottom(name_prefix + "c_" + tm1s);
      cont_c_param->add_bottom(name_prefix + "cont_" + ts);
      cont_c_param->add_top(name_prefix + "c_conted_" + tm1s);
    }
    // f_t (.) c_{t-1}
    {
      LayerParameter *f_c_tm1s_prod_parm = net_param->add_layer();
      f_c_tm1s_prod_parm->CopyFrom(prod_param);
      f_c_tm1s_prod_parm->set_name(name_prefix + "f_c_tm1s_prod_" + ts);
      f_c_tm1s_prod_parm->add_bottom(name_prefix + "f_" + ts);
      f_c_tm1s_prod_parm->add_bottom(name_prefix + "c_conted_" + tm1s);
      f_c_tm1s_prod_parm->add_top(name_prefix + "f_c_tm1s_prod_" + ts);
    }
    // i_t (.) g_t
    {
      LayerParameter *i_g_prod_parm = net_param->add_layer();
      i_g_prod_parm->CopyFrom(prod_param);
      i_g_prod_parm->set_name(name_prefix + "i_g_prod_" + ts);
      i_g_prod_parm->add_bottom(name_prefix + "i_" + ts);
      i_g_prod_parm->add_bottom(name_prefix + "g_" + ts);
      i_g_prod_parm->add_top(name_prefix + "i_g_prod_" + ts);
    }
    {
      LayerParameter *c_sum_layer = net_param->add_layer();
      c_sum_layer->CopyFrom(sum_param);
      c_sum_layer->set_name(name_prefix + "c_" + ts);
      c_sum_layer->add_bottom(name_prefix + "f_c_tm1s_prod_" + ts);
      c_sum_layer->add_bottom(name_prefix + "i_g_prod_" + ts);
      c_sum_layer->add_top(name_prefix + "c_" + ts);
    }
    {
      LayerParameter *H_c_param = net_param->add_layer();
      H_c_param->CopyFrom(H_activation_param);
      H_c_param->add_bottom(name_prefix + "c_" + ts);
      H_c_param->add_top(name_prefix + "H_c_" + ts);
      H_c_param->set_name(name_prefix + "H_c_" + ts);
    }
    {
      LayerParameter *h_parm = net_param->add_layer();
      h_parm->CopyFrom(prod_param);
      h_parm->set_name(name_prefix + "h_" + ts);
      h_parm->add_bottom(name_prefix + "o_" + ts);
      h_parm->add_bottom(name_prefix + "H_c_" + ts);
      h_parm->add_top(name_prefix + "h_" + ts);
    }
    output_concat_layer.add_bottom(name_prefix + "h_" + ts);
  } // for (int t = 1; t <= this->T_; ++t)

  {
    LayerParameter *c_T_copy_param = net_param->add_layer();
    c_T_copy_param->CopyFrom(split_param);
    c_T_copy_param->add_bottom(name_prefix + "c_" + format_int(this->T_));
    c_T_copy_param->add_top(name_prefix + "c_T");
  }
  net_param->add_layer()->CopyFrom(output_concat_layer);
}

template <typename Dtype>
void RNNv2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  T_ = bottom[0]->shape(0);
  N_ = bottom[0]->shape(1);
  LOG(INFO) << "Initializing recurrent layer: assuming input batch contains "
            << T_ << " timesteps of " << N_ << " independent streams.";

  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(T_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[1]->shape(1));

  activations_.clear();
  std::copy(this->layer_param_.recurrent_param().activations().begin(),
            this->layer_param_.recurrent_param().activations().end(),
            std::back_inserter(activations_));

  activation_alpha_.clear();
  std::copy(this->layer_param_.recurrent_param().activation_alpha().begin(),
            this->layer_param_.recurrent_param().activation_alpha().end(),
            std::back_inserter(activation_alpha_));

  activation_beta_.clear();
  std::copy(this->layer_param_.recurrent_param().activation_beta().begin(),
            this->layer_param_.recurrent_param().activation_beta().end(),
            std::back_inserter(activation_beta_));

  // Get (recurrent) input/output names.
  vector<string> output_names;
  OutputBlobNames(&output_names);
  vector<string> recur_input_names;
  RecurrentInputBlobNames(&recur_input_names);
  vector<string> recur_output_names;
  RecurrentOutputBlobNames(&recur_output_names);
  const int num_recur_blobs = recur_input_names.size();
  CHECK_EQ(num_recur_blobs, recur_output_names.size());

  // Create a NetParameter; setup the inputs that aren't unique to particular
  // recurrent architectures.
  NetParameter net_param;

  LayerParameter *input_layer_param = net_param.add_layer();
  input_layer_param->set_type("Input");
  InputParameter *input_param = input_layer_param->mutable_input_param();
  input_layer_param->add_top("x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  input_param->add_shape()->CopyFrom(input_shape);

  input_shape.Clear();
  for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    input_shape.add_dim(bottom[1]->shape(i));
  }
  input_layer_param->add_top("cont");
  input_param->add_shape()->CopyFrom(input_shape);

  // Call the child's FillUnrolledNet implementation to specify the unrolled
  // recurrent architecture.
  string direction = this->layer_param_.rnn_v2_param().direction();

  if(direction == "forward"){
    this->FillUnrolledNet(&net_param, "x", "cont", recur_input_names, output_names, recur_output_names, "");
  }
  else if (direction == "reverse"){
    {
      // reverse x
      LayerParameter *reverse_x_layer_param = net_param.add_layer();
      reverse_x_layer_param->set_type("Reverse");
      reverse_x_layer_param->add_bottom("x");
      reverse_x_layer_param->mutable_reverse_param()->add_axis(0);
      reverse_x_layer_param->add_top("x_rev");
      reverse_x_layer_param->set_name("x_rev");
    }
    this->FillUnrolledNet(&net_param, "x_rev", "cont", recur_input_names, output_names, recur_output_names, "bw_");
    {
      // reverse output back
      LayerParameter *reverse_output_layer_param = net_param.add_layer();
      reverse_output_layer_param->set_type("Reverse");
      reverse_output_layer_param->add_bottom("bw_" + output_names[0]);
      reverse_output_layer_param->mutable_reverse_param()->add_axis(0);
      reverse_output_layer_param->add_top(output_names[0]);
      reverse_output_layer_param->set_name(output_names[0]);
    }
    // copy and rename the recur_outputs
    {
      LayerParameter split_param;
      split_param.set_type("Split");
      LayerParameter *recur_output_copy_param;
      for(int i=0; i<num_recur_blobs; ++i){
        recur_output_copy_param = net_param.add_layer();
        recur_output_copy_param->CopyFrom(split_param);
        recur_output_copy_param->add_bottom("bw_" + recur_output_names[i]);
        recur_output_copy_param->add_top(recur_output_names[i]);
      }
    }

  } else if (direction == "bidirectional"){
    this->FillUnrolledNet(&net_param, "x", "cont", recur_input_names, output_names, recur_output_names, "fw_");
    {
      // reverse x
      LayerParameter *reverse_x_layer_param = net_param.add_layer();
      reverse_x_layer_param->set_type("Reverse");
      reverse_x_layer_param->add_bottom("x");
      reverse_x_layer_param->mutable_reverse_param()->add_axis(0);
      reverse_x_layer_param->add_top("x_rev");
      reverse_x_layer_param->set_name("x_rev");
    }
    this->FillUnrolledNet(&net_param, "x_rev", "cont", recur_input_names, output_names, recur_output_names, "bw_");
    {
      // reverse back the output of reverse direction
      LayerParameter *reverse_output_layer_param = net_param.add_layer();
      reverse_output_layer_param->set_type("Reverse");
      reverse_output_layer_param->add_bottom("bw_" + output_names[0]);
      reverse_output_layer_param->mutable_reverse_param()->add_axis(0);
      reverse_output_layer_param->add_top("bw_rev_"+output_names[0]);
      reverse_output_layer_param->set_name("bw_rev_"+output_names[0]);
    }
    // merge the outputs of forward and reverse
    {
      string merge_mode = this->layer_param_.rnn_v2_param().merge_mode();
      if (merge_mode == "concat"){
        LayerParameter output_concat_layer;
        output_concat_layer.set_name(output_names[0]);
        output_concat_layer.set_type("Concat");
        output_concat_layer.add_bottom("fw_" + output_names[0]);
        output_concat_layer.add_bottom("bw_rev_" + output_names[0]);
        output_concat_layer.add_top(output_names[0]);
        output_concat_layer.mutable_concat_param()->set_axis(-1);
      } else{
        LOG(ERROR) << "The value of merge_mode of RNNv2 layer is not supported: " << merge_mode;
        exit(-1);
      }
    }
    // concat on recur_outputs
    {
      LayerParameter concat_param;
      concat_param.set_type("Concat");
      concat_param.mutable_concat_param()->set_axis(0);
      LayerParameter *recur_output_param;
      for(int i=0; i<num_recur_blobs; ++i){
        recur_output_param = net_param.add_layer();
        recur_output_param->CopyFrom(concat_param);
        recur_output_param->add_bottom("fw_" + recur_output_names[i]);
        recur_output_param->add_bottom("bw_" + recur_output_names[i]);
        recur_output_param->add_top(recur_output_names[i]);
        recur_output_param->set_name(recur_output_names[i]);
      }
    }
  } else {
    LOG(ERROR) << "Wrong value of the direction parameter in RNNv2 layer";
    exit(-1);
  }

  // Prepend this layer's name to the names of each layer in the unrolled net.
  const string &layer_name = this->layer_param_.name();
  if (layer_name.size()) {
    for (int i = 0; i < net_param.layer_size(); ++i) {
      LayerParameter *layer = net_param.mutable_layer(i);
      layer->set_name(layer_name + "_" + layer->name());
    }
  }

  // Add "pseudo-losses" to all outputs to force backpropagation.
  // (Setting force_backward is too aggressive as we may not need to backprop to
  // all inputs, e.g., the sequence continuation indicators.)
  vector<string> pseudo_losses(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    LayerParameter* layer = net_param.add_layer();
    pseudo_losses[i] = output_names[i] + "_pseudoloss";
    layer->set_name(pseudo_losses[i]);
    layer->set_type("Reduction");
    layer->add_bottom(output_names[i]);
    layer->add_top(pseudo_losses[i]);
    layer->add_loss_weight(1);
  }


  // Create the unrolled net.
  unrolled_net_.reset(new Net<Dtype>(net_param));

  // Setup pointers to the inputs.
  x_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("x").get());
  cont_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("cont").get());

  // Setup pointers to paired recurrent inputs/outputs.
  recur_input_blobs_.resize(num_recur_blobs);
  recur_output_blobs_.resize(num_recur_blobs);
  for (int i = 0; i < recur_input_names.size(); ++i) {
    recur_input_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_input_names[i]).get());
    recur_output_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_output_names[i]).get());
  }

  // Setup pointers to outputs.
  output_blobs_.resize(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    output_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(output_names[i]).get());
  }

  // We should have 2 inputs (x and cont), plus a number of recurrent inputs
  CHECK_EQ(2 + num_recur_blobs, unrolled_net_->input_blobs().size());

  // This layer's parameters are any parameters in the layers of the unrolled
  // net. We only want one copy of each parameter, so check that the parameter
  // is "owned" by the layer, rather than shared with another.
  this->blobs_.clear();
  for (int i = 0; i < unrolled_net_->params().size(); ++i) {
    if (unrolled_net_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << i << ": "
                << unrolled_net_->param_display_names()[i];
      this->blobs_.push_back(unrolled_net_->params()[i]);
    }
  }
  // Check that param_propagate_down is set for all of the parameters in the
  // unrolled net; set param_propagate_down to true in this layer.
  for (int i = 0; i < unrolled_net_->layers().size(); ++i) {
    for (int j = 0; j < unrolled_net_->layers()[i]->blobs().size(); ++j) {
      CHECK(unrolled_net_->layers()[i]->param_propagate_down(j))
          << "param_propagate_down not set for layer " << i << ", param " << j;
    }
  }
  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Set the diffs of recurrent outputs to 0 -- we can't backpropagate across
  // batches.
  for (int i = 0; i < recur_output_blobs_.size(); ++i) {
    caffe_set(recur_output_blobs_[i]->count(), Dtype(0),
              recur_output_blobs_[i]->mutable_cpu_diff());
  }

  // Check that the last output_names.size() layers are the pseudo-losses;
  // set last_layer_index so that we don't actually run these layers.
  const vector<string> &layer_names = unrolled_net_->layer_names();
  last_layer_index_ = layer_names.size() - 1 - pseudo_losses.size();
  for (int i = last_layer_index_ + 1, j = 0; i < layer_names.size(); ++i, ++j) {
    CHECK_EQ(layer_names[i], pseudo_losses[j]);
  }
}

template <typename Dtype>
void RNNv2Layer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  CHECK_EQ(T_, bottom[0]->shape(0)) << "input number of timesteps changed";
  N_ = bottom[0]->shape(1);
  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(T_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[1]->shape(1));
  x_input_blob_->ReshapeLike(*bottom[0]);
  vector<int> cont_shape = bottom[1]->shape();
  cont_input_blob_->Reshape(cont_shape);
  vector<BlobShape> recur_input_shapes;
  RecurrentInputShapes(&recur_input_shapes);
  CHECK_EQ(recur_input_shapes.size(), recur_input_blobs_.size());
  for (int i = 0; i < recur_input_shapes.size(); ++i) {
    recur_input_blobs_[i]->Reshape(recur_input_shapes[i]);
  }
  unrolled_net_->Reshape();
  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);
  cont_input_blob_->ShareData(*bottom[1]);
  const int bottom_offset = 2;
  for (int i = bottom_offset, j = 0; i < bottom.size(); ++i, ++j) {
    CHECK(recur_input_blobs_[j]->shape() == bottom[i]->shape())
        << "shape mismatch - recur_input_blobs_[" << j
        << "]: " << recur_input_blobs_[j]->shape_string() << " vs. bottom[" << i
        << "]: " << bottom[i]->shape_string();
    recur_input_blobs_[j]->ShareData(*bottom[i]);
  }
  for (int i = 0; i < output_blobs_.size(); ++i) {
    top[i]->ReshapeLike(*output_blobs_[i]);
    top[i]->ShareData(*output_blobs_[i]);
    top[i]->ShareDiff(*output_blobs_[i]);
  }
  const int top_offset = output_blobs_.size();
  for (int i = top_offset, j = 0; i < top.size(); ++i, ++j) {
    top[i]->ReshapeLike(*recur_output_blobs_[j]);
  }
}

template <typename Dtype> void RNNv2Layer<Dtype>::Reset() {
  // "Reset" the hidden state of the net by zeroing out all recurrent outputs.
  for (int i = 0; i < recur_output_blobs_.size(); ++i) {
    caffe_set(recur_output_blobs_[i]->count(), Dtype(0),
              recur_output_blobs_[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
void RNNv2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  // Hacky fix for test time: reshare all the internal shared blobs, which may
  // currently point to a stale owner blob that was dropped when Solver::Test
  // called test_net->ShareTrainedLayersWith(net_.get()).
  // TODO: somehow make this work non-hackily.
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeights();
  }

  DCHECK_EQ(recur_input_blobs_.size(), recur_output_blobs_.size());

  unrolled_net_->ForwardTo(last_layer_index_);

  const int top_offset = output_blobs_.size();
  for (int i = top_offset, j = 0; i < top.size(); ++i, ++j) {
    top[i]->ShareData(*recur_output_blobs_[j]);
  }
}

INSTANTIATE_CLASS(RNNv2Layer);
REGISTER_LAYER_CLASS(RNNv2);
} // namespace caffe
