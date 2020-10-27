import os
import time
import torch
import random
import torch.nn as nn 
import argparse
from torch import optim 
import torchvision 
import torch.utils.data
import pandas as pd 
from model.ConvRRN import ST_EncDec as Model
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

if __name__ == '__main__': 

	parser = argparse.ArgumentParser(description='PCB anomaly detection')
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--num_instance', type=int, default=500) 
	parser.add_argument('--num_epochs', type=int, default=100) 
	parser.add_argument('--n_layers', type=int, default=2) 
	parser.add_argument('--lr', type=float, default=1e-4) 
	parser.add_argument('--init_tr', type=float, default=0.25) 
	parser.add_argument('--final_tr', type=float, default=0.0) 
	parser.add_argument('--gpu_ids', nargs='+', type=int, default=[1,2,3]) 
	parser.add_argument('--is_attn', action='store_true') 
	parser.add_argument('--mode', type=str, default='ConvConjLSTM', help='ConvLSTM, ConvConjLSTM') 
	parser.add_argument('--data', type=str, default='ekra') 
	# model load 
	args = parser.parse_args()

	root_dir = '%s/result'%(args.data) 
	if not os.path.isdir(root_dir): 
		os.mkdir(root_dir) 

	model_dir = root_dir + '/' + ((args.mode+'_a') if args.is_attn else args.mode)
	
	if not os.path.isdir(model_dir): 
		os.mkdir(model_dir) 

	model_file = '%s/%s'%(model_dir, 'model_dictionary.pt') 

	checkpoint = torch.load(model_file) 
	model = Model(args.mode, [1,64,64], [5,5], args.n_layers, args.is_attn)  
	model.load_state_dict(checkpoint['state_dict']) 

	if torch.cuda.device_count()>1: 
		if args.gpu_ids==None: 
			print("Let's use", torch.cuda.device_count(), "GPUs!") 
			device = torch.device('cuda:0') 
		else: 
			print("Let's use", len(args.gpu_ids), "GPUs!") 
			device = torch.device('cuda:' + str(args.gpu_ids[0])) 
	else: 
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
	
	model = torch.nn.DataParallel(model, device_ids=args.gpu_ids) 
	model = model.to(device) 

	for pb in [0.1, 0.2, 0.3, 0.4, 0.5]: 

		e_models = [] 
		e_gens = [] 
	
		e_models_true = [] 
		e_models_false = [] 
	
		e_gens_true = [] 
		
		x_normal = torch.load('%s/test_pcbs.pt'%(args.data))
		x_normal = x_normal.unsqueeze(1) # 5 by 1 by 20 by 1 by H by W 

		mus = [-5,-4,-3,-2,2,3,4,5] 
	
		pcb_info = torch.load('%s/pcb_info.pt'%(args.data)) 
		df_info = pd.read_csv('%s/df_info.csv'%(args.data)) 
	
		height = pcb_info['height'] 
		width = pcb_info['width'] 
		mean = pcb_info['mean'] 
		std = pcb_info['std'] 
	
		x_ = df_info[df_info.Timestep==0]['X'] 
		y_ = df_info[df_info.Timestep==0]['Y'] 
		
		mask = torch.zeros(1, height, width) 
		for x,y in zip(x_, y_): 
			mask[0, int(y), int(x)] = 1 
	
		mask = mask.view(1,1,1,*mask.size()).expand_as(x_normal) 
	
		threshold = list(np.linspace(0,max(mus),50+1)) 
	
		tp_e, fp_e, fn_e, tn_e = {}, {}, {}, {} 
		tp_i, fp_i, fn_i, tn_i = {}, {}, {}, {} 
	
		for th in threshold: 
			tp_e[th] = 0 
			fp_e[th] = 0
			fn_e[th] = 0
			tn_e[th] = 0 
	
			tp_i[th] = 0 
			fp_i[th] = 0
			fn_i[th] = 0
			tn_i[th] = 0 
	
	
		model.eval() 
	
		for mu in mus: 
			
			mask_a = mask.new(*mask.size()).bernoulli_(pb) 
			e_gen = mask_a*mask*(mu + 0.1*torch.randn(x_normal.size())) # torch.randn(x_normal.size()))
	
			idx = mask==1
			idx_true = idx & (mask_a==1) 
			idx_false = idx & (mask_a==0) 
	
			e_gens.append(e_gen[idx])
			e_gens_true.append(e_gen[idx_true]) 
	
			x_in = (x_normal + e_gen).clone() 
		
			e_model = []
			e_model_true = [] 
			e_model_false = [] 
	
			for t in range(len(x_in)): 
				inputs = x_in[t].to(device) 
				outputs = model(inputs) 
				err = inputs-outputs 
	#			err = inputs.clone() 
	
				err = err.cpu().detach() 
				e_model.append(err[idx[t]]) 
	
				for th in threshold: 
	
					excessive_val = (e_gen[t][idx[t]]>0).int()*2 + (err[idx[t]]>th).int()
					insufficient_val = (e_gen[t][idx[t]]<0).int()*2 + (err[idx[t]]<-th).int() 
	
					tp_e[th]+=(excessive_val==3).sum().item() 
					fp_e[th]+=(excessive_val==1).sum().item() 
					fn_e[th]+=(excessive_val==2).sum().item() 
					tn_e[th]+=(excessive_val==0).sum().item() 
	
					tp_i[th]+=(insufficient_val==3).sum().item() 
					fp_i[th]+=(insufficient_val==1).sum().item() 
					fn_i[th]+=(insufficient_val==2).sum().item() 
					tn_i[th]+=(insufficient_val==0).sum().item() 
	
				e_model_true.append(err[idx_true[t]]) 
				e_model_false.append(err[idx_false[t]]) 	
				
			e_model = torch.cat(e_model, 0) 
			e_models.append(e_model) 
	
			e_model_true = torch.cat(e_model_true, 0) 
			e_models_true.append(e_model_true) 
	
			e_model_false = torch.cat(e_model_false, 0) 
			e_models_false.append(e_model_false) 
	
	
		precision_e, recall_e, f1_e = [], [], [] 
		precision_i, recall_i, f1_i = [], [], []  
	
		for th in threshold:
			p_e = tp_e[th]/(tp_e[th]+fp_e[th]+1e-7)
			r_e = tp_e[th]/(tp_e[th]+fn_e[th]+1e-7)
	
			p_i = tp_i[th]/(tp_i[th]+fp_i[th]+1e-7)
			r_i = tp_i[th]/(tp_i[th]+fn_i[th]+1e-7)
	
			if p_e!=0 and r_e!=0: 
				precision_e.append(p_e) 
				recall_e.append(r_e) 
				f1 = 2/(1/p_e + 1/r_e) 
				f1_e.append(f1) 
	
			if p_i!=0 and r_i!=0: 
				precision_i.append(p_i) 
				recall_i.append(r_i)
				f1 = 2/(1/p_i + 1/r_i) 
				f1_i.append(f1) 
	
		# draw precision-recall curve 
		pr_info_e = {'Precision': precision_e, 'Recall': recall_e, 'F1': f1_e}
		pr_info_i = {'Precision': precision_i, 'Recall': recall_i, 'F1': f1_i}
	
		pr_save_file = '%s/%s'%(model_dir, 'pr_info_e_%d.pt'%(pb*10)) 
		torch.save(pr_info_e, pr_save_file) 
	
		pr_save_file = '%s/%s'%(model_dir, 'pr_info_i_%d.pt'%(pb*10)) 
		torch.save(pr_info_i, pr_save_file) 

#	sns.set(color_codes=True) 
#	fig1 = plt.subplots(nrows=4, ncols=4, figsize=(15,9)) 
#	plt.subplots_adjust(hspace=0.1) 
#
#	for  i, idx in enumerate([0,2,5,7]): 
#
#		plt.subplot(4,4,i+1) 
#		ax = sns.distplot(e_gens[idx], color='gray')#, bins=20)
#		ax.set_xlim(-10,10)
#		ax.set_ylim(0,1) 
#		plt.xticks(fontsize=8) 
#		plt.yticks(fontsize=8) 
#
#		plt.subplot(4,4,4+i+1) 
#		ax = sns.distplot(e_gens_true[idx], color='red') 
#		ax.set_xlim(-10,10) 
#		ax.set_ylim(0,1) 
#		plt.xticks(fontsize=8) 
#		plt.yticks(fontsize=8) 
#
#		plt.subplot(4,4,4*2+i+1) 
#		ax = sns.distplot(e_models[idx], color='gray')#, bins=20) 
#		ax.set_xlim(-10,10)
#		ax.set_ylim(0,1) 
#		plt.xticks(fontsize=8) 
#		plt.yticks(fontsize=8)
#
#		plt.subplot(4,4,4*3+i+1) 
#		ax = sns.distplot(e_models_false[i], color='deepskyblue')#, bins=20) 
#		sns.distplot(e_models_true[idx], color='red')#, bins=20) 
#		ax.set_xlim(-10,10) 
#		ax.set_ylim(0,1) 
#
#		plt.xticks(fontsize=8) 
#		plt.yticks(fontsize=8)
#					
#
#
#	dist_save_file = '%s/%s'%(model_dir, 'error.png') 
#	plt.savefig(dist_save_file) 
#
