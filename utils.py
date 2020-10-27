import os
import numpy as np
import torch
import h5py
from matplotlib import cm 

def make_colormaps(images, limit=1.0): 
	""" input argments: 
	- images: anomaly maps 
		(# of images) by (height) by (width) 
	- limit: lower or upper bound 
		the magnitudes of the lower and upper bound set equally 

	output arguments: 
	- scaled: anomaly maps painted as red(excessive) or blue(insufficient) colors 
		(# of images) by (rgb channel) by (height) by (width) 
		if masks is not None, the pad information is also displayed 
	"""

	images = images.double()  
	images.clamp_(-limit, limit) 

	scaled = (images+limit)/(2*limit) 
	scaled = cm.bwr(scaled)    

	scaled = torch.tensor(scaled)[:,:,:,:3]
	scaled = scaled.permute(0, 3, 1, 2) 

	return scaled 
