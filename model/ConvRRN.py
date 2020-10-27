import torch
import torch.nn as nn 
from model.core_module.ConvLSTM import ConvLSTM
from model.SpatialAttention import SpatialAttention 

class StackedConvLSTM(nn.Module):
	def __init__(self, mode, channel, kernel, n_layers, is_attn=False): 
		super(StackedConvLSTM, self).__init__() 
		self.channel = channel 
		self.kernel = kernel
		self.n_layers = n_layers 
		self.is_attn = is_attn	

		if mode=='ConvLSTM': 
			from model.core_module.ConvLSTM import ConvLSTM 	
		elif mode=='ConvConjLSTM': 
			from model.core_module.ConvConjLSTM import ConvConjLSTM as ConvLSTM 
		elif mode=='ConvSTLSTM': 
			from model.core_module.ConvSTLSTM import ConvSTLSTM as ConvLSTM 
		elif mode=='ConvGridLSTM': 
			from model.core_module.ConvGridLSTM import ConvGridLSTM as ConvLSTM 


		conv_lstm = [ConvLSTM(channel, kernel, 2)]
		for _ in range(1, n_layers):
			conv_lstm.append(ConvLSTM(channel, kernel))  

		self.conv_lstm = nn.Sequential(*conv_lstm) 

		if is_attn: 
			s_attn = [SpatialAttention(channel, kernel) for _ in range(n_layers)] 
			self.s_attn = nn.Sequential(*s_attn) 
	
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



class S_EncDec(nn.Module): 
	def __init__(self, channels, kernels, encoder=True): 

		super(S_EncDec, self).__init__()
		layer = [] 		

		if not encoder: 
			channels.reverse() 
			kernels.reverse() 

		for i in range(len(kernels)):
		
			if encoder: 
				conv = nn.Conv2d(channels[i], channels[i+1], kernels[i], 2, bias=True) 
				layer.append(conv) 
				layer.append(nn.ReLU())
				layer.append(nn.BatchNorm2d(channels[i+1])) 

			if not encoder: 
				conv = nn.ConvTranspose2d(channels[i], channels[i+1], kernels[i], 2, bias=True) 
				layer.append(conv) 
				if i<len(kernels)-1: 
					layer.append(nn.ReLU())
					layer.append(nn.BatchNorm2d(channels[i+1])) 

		self.layer = nn.Sequential(*layer) 

	def forward(self, inputs): 

		inputs_size = inputs.size() 
		inputs = inputs.view(-1, *inputs_size[-3:])
		outputs = self.layer(inputs) 
		return outputs.view(*inputs_size[:-3], *outputs.size()[-3:]) 


class ST_Encoder(nn.Module): 
	def __init__(self, mode, channels, kernels, n_layers, is_attn): 
	
		super(ST_Encoder, self).__init__() 

		self.channels = channels 
		self.kernels = kernels 
		self.s_enc = S_EncDec(channels[:-1], kernels[:-1], encoder=True) 
		self.conv_lstm = StackedConvLSTM(mode, channels[-1], kernels[-1], n_layers, is_attn) 

	def forward(self, inputs): 
	
		inputs = self.s_enc(inputs) 
		hidden, attns = self.conv_lstm(inputs) 

		return hidden, attns 


class ST_Decoder(nn.Module): 
	def __init__(self, mode, channels, kernels, n_layers): 
	
		super(ST_Decoder, self).__init__() 

		self.channels = channels 
		self.kernels = kernels
		self.s_enc = S_EncDec(channels[:-1], kernels[:-1], encoder=True) 
		self.conv_lstm = StackedConvLSTM(mode, channels[-1], kernels[-1], n_layers) 
		self.s_dec = S_EncDec(channels, kernels, encoder=False) 

	def forward(self, hiddens, targets, attns=None, teacher_ratio=1.0): 

		steps = targets.size(1) 
		timesteps = range(steps-1, 0, -1) 

		if attns is not None:
			for i in range(len(attns)): 
				hiddens[i][0] = hiddens[i][0]-(attns[i][-1].expand_as(hiddens[i][0])) 

		outputs = [self.s_dec(hiddens[-1][0])]

		for step in timesteps:

			if self.training: 
				x = targets[:, step] 
				teacher_mask = x.new(x.size(0), 1, 1, 1).bernoulli_(teacher_ratio) 
				teacher_mask = teacher_mask.expand_as(x) 
				x = teacher_mask*x + (1-teacher_mask)*outputs[-1] 
			else: 				
				x = outputs[-1] 

			x = self.s_enc(x) 
			hiddens, _ = self.conv_lstm(x[:,None], hiddens) 
			if attns is not None: 
				for i in range(len(attns)): 
					hiddens[i][0] = hiddens[i][0]-(attns[i][step-1].expand_as(hiddens[i][0])) 

			outputs.append(self.s_dec(hiddens[-1][0])) 

		outputs = torch.stack(outputs[::-1], 1) 

		return outputs 
		
	
class ST_EncDec(nn.Module): 
	def __init__(self, mode, channels, kernels, n_layers, is_attn=False): 
		super(ST_EncDec, self).__init__() 

		self.st_encoder = ST_Encoder(mode, channels, kernels, n_layers, is_attn) 
		self.st_decoder = ST_Decoder(mode, channels, kernels, n_layers) 

	def forward(self, inputs, teacher_ratio=1.0): 

		hidden, attns = self.st_encoder(inputs) 
		outputs = self.st_decoder(hidden, inputs, attns, teacher_ratio=teacher_ratio)

		return outputs
   		


