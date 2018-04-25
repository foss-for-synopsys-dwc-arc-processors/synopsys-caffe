A Short Summary of New Features in Synopsys Caffe
=================================================

Synopsys Caffe Version: 2018.06  
New added features are compared with the original BVLC Caffe 1.0.0

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

TensorFlow (Converter) related
------------------------------
AVE_TF (average pooling excluding the paddings) in PoolingParameter  
*pad_type (asymmetric padding) in ConvolutionParameter and PoolingParameter  
ceil_mode in PoolingParameter  
relu6 in ReLUParameter  
pad.py (customized Python layer)  
stridedslice.py (customized Python layer)  
shape.py (customized Python layer)  
  

***Special Note**: pad_type (asymmetric padding) does not support the cuDNN acceleration now, [explanations](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/commit/b193bc72180a295ea9322837a9735cd72a552f7e#comments); To use it, you must turn off the cuDNN acceleration switch in *Makefile.config* for building.
  
  
For more details, please refer to [caffe.proto](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/blob/master/src/caffe/proto/caffe.proto) and the corresponding source code.

