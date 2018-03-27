A Short Summary of New Features in Synopsys Caffe
=================================================

Synopsys Caffe Version: 2018.03  
New added features are compared with the original BVLC Caffe 1.0.0

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
  

__*Special Note__: pad_type (asymmetric padding) does not support the cuDNN acceleration now; To use it, you must turn off the cuDNN acceleration switch in _Makefile.config_ for building.


For more details, please refer to the _caffe.proto_ and the corresponding source code.

