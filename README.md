# CRRN (Convolutional Reconstructive Recurrent Network)

## CRRN architecture
![Screenshot from 2020-10-27 16-46-18](https://user-images.githubusercontent.com/43340417/97271493-f83d3a80-1873-11eb-8c01-f0afc3c8ec69.jpg)

## Usage
* Clone this repo:
```git clone https://github.com/kimwin2/CRRN_AnomalyDetection.git```
* For training: 
```
python3 0_preprocessing.py
python3 1_train.py --batch_size 5 --gpu_ids 0 --mode ConvConjLSTM
```
* For testing: 
```
python3 2_test_model.py --batch_size 5 --gpu_ids 0 --mode ConvConjLSTM
```


## network files

* ConvRRN.py: It is a wrapper class that adds functional things to use cstm.
* SpatialAttention.py: This code is for spatial attention in the encoder-decoder structure.
* core_module/ConvConjLSTM.py: This code is a cstm(convolutional spatio-temporal memory) structure.
* core_module/ConvSTLSTM.py: This code is a convolutional spatio-temporal LSTM structure.
* core_module/ConvLSTM.py: This code is a convolutional LSTM structure.

## cstm(convolutional spatio-temporal memory) example code
* location of cstm: model/core_module/ConvConjLSTM.py
* code example
```
from model.core_module.ConvConjLSTM import ConvConjLSTM as ConvLSTM

class StackedConvLSTM(nn.Module):
	def __init__(self, mode, channel, kernel, n_layers, is_attn=False): 
    self.channel = channel 
		self.kernel = kernel
		self.n_layers = n_layers 
		self.is_attn = is_attn	
    
    conv_lstm = [ConvLSTM(channel, kernel, 2)]
    for _ in range(1, n_layers):
      conv_lstm.append(ConvLSTM(channel, kernel))  
    self.conv_lstm = nn.Sequential(*conv_lstm) 

	def forward(self, inputs, hiddens=None): 
		steps = inputs.size(1)
		if hiddens is None: 
			hiddens = [None for i in range(self.n_layers)] 

		xm = [[inputs[:,j], None] for j in range(steps)]

		attns = None 
		if self.is_attn: 
			attns = [] 

		for i in range(self.n_layers): 
	
			ym = [] 		
			if self.is_attn: 
				attn = [] 
			h_mask = None 
			for j in range(steps): 
				h, c = self.conv_lstm[i](xm[j], hiddens[i]) 

				if type(h) is tuple: 
					h_up, h = h
				else: 
					h_up = h 

				if type(c) is tuple: 
					c, m = c 
				else:
					m = c

				if self.is_attn: 
					h, a = self.s_attn[i](h)
					attn.append(a) 

				if self.training: 
					if h_mask is None: 
						h_mask = h.new(*h.size()).bernoulli_(0.8).div(0.8) 
						h = h*h_mask 
				
				ym.append([h_up,m]) 
				hiddens[i] = [h,c]

			if self.is_attn:
				attns.append(attn) 

			xm = ym 
	
		return hiddens, attns 	

class wrapper_class(nn.Module):
  def __init__(self, mode, channels, kernels, n_layers, is_attn):
    super(wrapper_class, self).__init__()
    
    self.channels = channels 
		self.kernels = kernels 
    self.conv_lstm = StackedConvLSTM(mode, channels[-1], kernels[-1], n_layers, is_attn) 
  
  def forward(self, inputs):  
		hidden, attns = self.conv_lstm(inputs) 

		return hidden, attns 
```
* detailed code and information : model/ConvRRN.py

## paper

Y.-H. Yoo, U.-H. Kim and J.-H. Kim, "Convolutional Recurrent Reconstructive Network for Spatiotemporal Anomaly Detection in Solder Paste Inspection," IEEE Trans. on Cybernetics, Accepted, Oct. 2020.


