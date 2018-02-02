import caffe
import numpy as np


class TensorSlice(caffe.Layer):
    """
    Get a tensor's slicing: realize the function of the tf.strided_slice().

    TO OPTIMIZE: For simplification, assume all the input tensors are 4 dimensions now.
    """

    def setup(self, bottom, top):
        # check number of inputs and outputs
        if len(bottom) != 1:
            raise Exception("Only input one Tensor at a time!")
        if len(top) != 1:
	    raise Exception("Only output one Tensor at a time!")

	params = eval(self.param_str)
	self.begin = np.array(params["begins"])
	self.end = np.array(params["ends"])
	if params["strides"] != None:
	  self.strides = np.array(params["strides"])
	else:
	  self.strides = np.array([1, 1, 1, 1])
	
	if params["beginmask"] != None:
	  self.beginmask = int(params["beginmask"])
	else:
	  self.beginmask = None
	if params["endmask"] != None:
	  self.endmask = int(params["endmask"])
	else:
	  self.endmask = None

	# Handles the condition where the "ends" is assigned (0, 0, 0, 0) meaning the end of the input Tensor 
	if ((self.end==0).all() and (self.strides>0).all()):
	  self.end = bottom[0].shape
	
	# According to the tf.strided_slice():
	# If the ith bit of beginmask is set, begin[i] is ignored and the fullest possible range in that dimension is used instead;
	# The endmask works analogously, except with the end range.
	if self.beginmask != None:
	  self.begin[self.beginmask] = 0
	if self.endmask != None:
	  self.end[self.endmask] = bottom[0].shape[self.endmask]
	
	# other parameters...


    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count == 0:
            raise Exception("Input must not be empty!")
	
	#top[0].reshape(*bottom[0].data.shape)		
	#dim = len(self.begin) 
	num = [0, 0, 0, 0]
	for i in range(4):
	  if self.strides[i]==0:
	    raise Exception("Strides should never equal to 0!")
	  else: 
	    num[i] = (abs(self.end[i]-self.begin[i])/abs(self.strides[i]))	
	top[0].reshape(num[0], num[1], num[2], num[3])


    def forward(self, bottom, top):
	#pass
	#top[0].data[...] = bottom[0].data[:]
	top[0].data[...] = np.zeros(top[0].shape)
	for i in range(len(self.begin)):
       	  top[0].data[...] = bottom[0].data[self.begin[0]:self.end[0]:self.strides[0], self.begin[1]:self.end[1]:self.strides[1], self.begin[2]:self.end[2]:self.strides[2], self.begin[3]:self.end[3]:self.strides[3]]


    def backward(self, top, propagate_down, bottom):
	for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[i].diff[:]

