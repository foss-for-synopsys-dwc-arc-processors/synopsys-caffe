A Short Summary of New Features in Synopsys Caffe
=================================================

Synopsys Caffe Version: 2018.09  
New added features are compared with the original BVLC Caffe 1.0.0
  
evconvert (TensorFlow/ONNX/... to Caffe Converter) related  
----------------------------------------------------------
pad_type (deprecated, "SAME" style padding) in ConvolutionParameter and PoolingParameter  
pad_l, pad_r, pad_t and pad_b (arbitrary 2D padding) in ConvolutionParameter and PoolingParameter  
AVE_EXC_PAD (average pooling excluding the paddings), AVE_TF (deprecated, alias for AVE_EXC_PAD) in PoolingParameter  
ceil_mode in PoolingParameter  
relu6 in ReLUParameter  
pad.py (customized Python layer, implementation of tf.pad)  
stridedslice.py (deprecated, customized Python layer)  
slice.py (customized Python layer, implementation of tf.slice and tf.strided_slice)  
shape.py (customized Python layer, implementation of tf.shape)  
stack.py (customized Python layer, implementation of tf.stack)  
eltwise.py (customized Python layer, functional expansion of ELtwise layer)  
  
evprune (Network Pruning Tool) related  
--------------------------------------
squeeze_conv_layer  
squeeze_inner_product_layer  
squeeze_deconv_layer  
  
evquantize related (only valid for CUDA forwards implementation)  
----------------------------------------------------------------------------------  
input_scale, output_scale in ConvolutionParameter, InnerProductParameter and LRNParameter  
output_scale in EltwiseParameter  
input_scale in SoftmaxParameter  
output_shift_instead_division in PoolingParameter  
saturate in ReLUParameter and PoolingParameter  
  
YOLO related  
--------------  
add_eps_before_sqrt in BatchNormParameter, MVNParameter and NormalizeParameter  
caffe_yolo in TransformationParameter  
jitter in ResizeParameter  
exposure_lower, exposure_upper in DistortionParameter  
side in DataParameter  
reorg_layer  
yolo_v2_loss_layer  
yolo_v3_loss_layer  
    
ICNet (PSPNet) related  
---------------------  
adaptive_bias_channel_layer  
bias_channel_layer  
cudnn_bn_layer  
densecrf_layer  
domain_transform_forward_only_layer  
domain_transform_layer  
image_seg_data_layer  
interp_layer  
mat_read_layer  
mat_write_layer  
seg_accuracy_layer  
spatial_product_layer  
unique_label_layer  
bn_layer (slope_filler, bias_filler, momentum and icnet in BNParameter)  
update_global_stats and icnet in BatchNormParameter  
scale_factors, crop_width and crop_height in TransformationParameter  
      
SRGAN related  
-------------
gan_loss_layer  
gan_solver in SolverParameter  
pixelshuffler in ReshapeParameter  
dis_mode and gen_mode in ConvolutionParameter, InnerProductParameter and ScaleParameter  
weight_fixed in ConvolutionParameter and InnerProductParameter  
  
FlowNet2 related
----------------
accum_layer  
augmentation_layer  
black_augmentation_layer  
channel_norm_layer  
correlation_1d_layer  
correlation_layer  
custom_data_layer  
data_augmentation_layer  
downsample_layer  
flo_writer_layer  
float_reader_layer  
float_writer_layer  
flow_augmentation_layer  
flow_warp_layer  
generate_augmentation_parameters_layer  
l1_loss_layer  
lpq_loss_layer  
mean_layer  
resample_layer  
  
SSD related
-----------
annotated_data_layer    
detection_evaluate_layer  
detection_output_layer  
multibox_loss_layer  
normalize_layer  
permute_layer  
prior_box_layer  
smooth_L1_loss_layer  
video_data_layer  
  
Faster RCNN related
-------------------
roi_pooling_layer  
smooth_L1_loss_layer (sigma and abssum in SmoothL1LossParameter)  
scale_train in DropoutParameter  
  
SegNet related
--------------
bn_layer  
dense_image_data_layer  
upsample_layer  
sample_weights_test in DropoutParameter  
weight_by_label_freqs in LossParameter  
    
    
For more details, please refer to [caffe.proto](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/blob/master/src/caffe/proto/caffe.proto) and the corresponding source code.

