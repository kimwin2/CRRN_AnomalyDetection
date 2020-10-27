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

class dataset_pcb(torch.utils.data.Dataset):
	def __init__(self, data_name, num_instance=500): 
		super(dataset_pcb, self).__init__() 
		
		self.data_name = data_name
		self.num_instance = num_instance 

		pcb_info = torch.load('%s/pcb_info.pt'%(data_name)) 
		df_info = pd.read_csv('%s/df_info.csv'%(data_name)) 

		self.height = pcb_info['height'] 
		self.width = pcb_info['width'] 
		self.mean = pcb_info['mean']
		self.std = pcb_info['std']

		x_ = df_info[df_info.Timestep==0]['X'] 
		y_ = df_info[df_info.Timestep==0]['Y'] 

		mask = torch.zeros(1, self.height, self.width) 

		for x,y in zip(x_, y_):
			mask[0, int(y), int(x)] = 1
	
		self.mask = mask 
		self.data = torch.load('%s/test_pcbs.pt'%(data_name)) 

#		print(self.data.size()) 
	def __getitem__(self, index): 
		return self.data[index] 	

	def __len__(self): 
		return len(self.data) 

	def gen_each_pcb(self): 

		ideal_pcbs = [] 
		noisy_pcbs = [] 

		for i in range(len(self.mean)): 
			ideal_pcb = (torch.randn(1, self.height, self.width) * self.std[i]) + self.mean[i]
			ideal_pcb *= self.mask

			# add noisy 
			mask_a = self.mask.new(*self.mask.size()).bernoulli_(0.1) 
			anomaly_pcb = mask_a*self.mask*(torch.randn(self.mask.size()))
			
			noisy_pcb = ideal_pcb.detach() + anomaly_pcb.detach()

			ideal_pcbs.append(ideal_pcb) 
			noisy_pcbs.append(noisy_pcb) 

		ideal_pcbs = torch.stack(ideal_pcbs, 0) 
		noisy_pcbs = torch.stack(noisy_pcbs, 0) 

		return ideal_pcbs, noisy_pcbs 

	def gen_pcb(self, num_instance):
		
		ideal_pcbs, noisy_pcbs = [], [] 
		for i in range(num_instance): 
			ideal_pcb, noisy_pcb = self.gen_each_pcb() 
			ideal_pcbs.append(ideal_pcb) 
			noisy_pcbs.append(noisy_pcb) 

		ideal_pcbs = torch.stack(ideal_pcbs, 0) 
		noisy_pcbs = torch.stack(noisy_pcbs, 0) 

		return ideal_pcbs, noisy_pcbs 

if __name__ == '__main__': 

	parser = argparse.ArgumentParser(description='PCB anomaly detection')
	parser.add_argument('--batch_size', type=int, default=5)
	parser.add_argument('--num_instance', type=int, default=500) 
	parser.add_argument('--num_epochs', type=int, default=100) 
	parser.add_argument('--n_layers', type=int, default=2) 
	parser.add_argument('--lr', type=float, default=1e-4) 
	parser.add_argument('--init_tr', type=float, default=0.25) 
	parser.add_argument('--final_tr', type=float, default=0.0) 
	parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1,2,3]) 
	parser.add_argument('--is_attn', action='store_true') 
	parser.add_argument('--data', type=str, default='ekra') 
	parser.add_argument('--mode', type=str, default='ConvConjLSTM', help='ConvLSTM, ConvConjLSTM') 

	args = parser.parse_args()

	root_dir = '%s/result'%(args.data) 
	if not os.path.isdir(root_dir): 
		os.mkdir(root_dir) 

	model_dir = root_dir + '/' + ((args.mode+'_a') if args.is_attn else args.mode)
	
	if not os.path.isdir(model_dir): 
		os.mkdir(model_dir) 

	model_file = '%s/%s'%(model_dir, 'model_dictionary.pt') 

	dataset = dataset_pcb(args.data, num_instance = args.num_instance)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=4) 

	model = Model(args.mode, [1,64, 64], [5,5], args.n_layers, args.is_attn)  
	criterion = nn.MSELoss() 

	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) 

	if os.path.isfile(model_file): 
		print('Load saved model') 
		checkpoint = torch.load(model_file) 
		init_epoch = checkpoint['epoch'] 
		test_loss = checkpoint['test_loss'] 
		print(test_loss) 
		model.load_state_dict(checkpoint['state_dict']) 
		optimizer.load_state_dict(checkpoint['optimizer']) 
	else: 
		init_epoch = -1
		test_loss = [] 

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

	for state in optimizer.state.values(): 
		for k,v in state.items():
			if isinstance(v, torch.Tensor): 
				state[k] = v.to(device) 

	for epoch in range(init_epoch+1, args.num_epochs): 

		start_time = time.time() 
		each_train_loss = [] 
		model.train()

		b = args.init_tr 
		a = (args.final_tr-args.init_tr)/(args.num_epochs-1.0) 
		teacher_ratio = a*epoch + b	

		with torch.set_grad_enabled(True): 
			for i in range(args.num_instance//args.batch_size): 
				targets, inputs = dataset.gen_pcb(num_instance=args.batch_size)

				mask_denoise = inputs.new(inputs.size(0), *inputs.size()[-3:]).bernoulli_(0.9)/0.9
				inputs*= mask_denoise.unsqueeze(1).expand_as(inputs) 			

				iter_time = time.time() 

				inputs = inputs.to(device) 
				targets = targets.to(device) 

				optimizer.zero_grad() 
				outputs = model(inputs, teacher_ratio=teacher_ratio)
				
				err = criterion(outputs, targets)
				err.backward() 

				torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
				optimizer.step() 

				print('iter_time = ', (time.time() - iter_time)/args.batch_size) 
				assert(False) 

				each_train_loss.append(err.item()) 

		each_test_loss = [] 
		model.eval() 

		with torch.set_grad_enabled(False): 
			for inputs in dataloader:
				inputs = inputs.to(device) 
				outputs = model(inputs)
	
				err = criterion(outputs, inputs)
				each_test_loss.append(err.item()) 

		epoch_train_loss = sum(each_train_loss)/len(each_train_loss)  
		epoch_test_loss = sum(each_test_loss)/len(each_test_loss) 

		test_loss.append(epoch_test_loss) 

		print(epoch, 'train: ', epoch_train_loss, ', test: ', epoch_test_loss, ', elapsed: %4.2f'%(time.time()-start_time)) 

		model_dictionary = {'epoch': epoch, 
			'test_loss': test_loss, 
			'state_dict': list(model.children())[0].state_dict(),
			'optimizer': optimizer.state_dict()
		} 

		torch.save(model_dictionary, model_file)
