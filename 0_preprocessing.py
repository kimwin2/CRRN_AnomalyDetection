import os
import torch 
import json
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import norm 
import argparse 

def autocorr1(x, lags): 
	corr = [1. if l==0 else np.corrcoef(x[:-l], x[l:])[0][1] for l in lags] 
	return np.array(corr)

if __name__ =='__main__': 

	parser = argparse.ArgumentParser(description='PCB anomaly detection')
	parser.add_argument('--data', type=str, default='ekra') 
	args = parser.parse_args() 

	if args.data=='jabil':
		cycle = 10 
	else: 
		cycle = 20 
	
	job = json.loads(open('%s/job.json'%(args.data)).read()) 
	
	width = int(job['PCBInfo']['PCBLenth'][0]+0.5)
	height = int(job['PCBInfo']['PCBWidth'][0]+0.5)
	
	if width%4==0: 
		width += 1
	else:
		width = (width+4)//4*4 + 1 
	
	if height%4==0: 
		height += 1
	else: 
		height = (height+4)//4*4 + 1 
	
	# get group
	shape = np.zeros(len(job['Pad']['ID'])) 
	
	for i in range(len(job['Pad']['ID'])): 	
		idd = int(job['Pad']['ID'][i])
		idd_job = int(job['Pad']['ShapeID'][i]) 
		shape[idd-1] = idd_job 
	
	timestep = [] 
	posx = [] 
	posy = []
	volume = [] 
	padid = [] 
	group = [] 
	shape_type = [] 
	all_timestep = 0 
		
	for i in range(len(os.listdir('%s/spi_measurement'%(args.data)))): 
		
		result_file = '%s/spi_measurement/spi_%03d.json'%(args.data, i) 
		result = json.loads(open(result_file).read()) 
#		result = json.loads(open('%s/spi_measurement/%s'%(args.data, result_file)).read()) 
		pad_number = len(result['PADINFO']) 
	
		timestep += [i]*(pad_number) 
		all_timestep+=1 	
	
		for j in range(pad_number): 
			x = result['PADINFO'][j]['PosX'] 
			y = result['PADINFO'][j]['PosY'] 
			vol = result['PADINFO'][j]['Volume'] 
	
			pid = int(result['PADINFO'][j]['PadID'])-1 
			gid = shape[pid] 
			shape_type.append(gid) 
	
			posx.append(x) 
			posy.append(y)
			volume.append(vol)
			padid.append(pid) 
			group.append(gid) 

	
	shape_type = list(set(shape_type)) 
	result = {'Timestep': timestep, 'PadID': padid, 'X': posx, 'Y': posy, 'Vol': volume, 'Group': group} 
	
	df = pd.DataFrame(data=result)
	
	norm_vol = [] 
	for i in range(all_timestep): 
		norm_vol.append(list(df[df.Timestep==i]['Vol'])) 
	
	norm_vol = np.array(norm_vol) 
	pcb_mean = norm_vol.mean(0) 
	pcb_std = norm_vol.std(0) 
	norm_vol = (norm_vol-norm_vol.mean(0))/(norm_vol.std(0)) 
	
	df['Vol_norm'] = norm_vol.flatten() 
	df['Age'] = df['Timestep']%cycle 
	
	# plot 
	fig = plt.figure(figsize=(7,2.5)) 
	plt.subplots_adjust(wspace=0.1) 
	sns.set(font_scale=0.8) 

	gs = gridspec.GridSpec(1,2,width_ratios=[4,1]) 
	plt.subplot(gs[0]) 
	raw_line_plot = sns.lineplot(x='Timestep', 
		y='Vol', data=df[(df.PadID==100) | (df.PadID==300) | (df.PadID==500)], 
		hue='PadID', style='PadID', palette=['b','r','g'], legend=False) 
	plt.xlabel('Timestep') 
	plt.ylabel('Volume') 
	
	plt.subplot(gs[1]) 
	raw_dist_plot1 = sns.distplot(df[df.PadID==100]['Vol'], fit=norm, kde=False, bins=20, vertical=True, color='#0934c9') 
	raw_dist_plot2 = sns.distplot(df[df.PadID==300]['Vol'], fit=norm, kde=False, bins=20, vertical=True, color='#ff0000') 
	raw_dist_plot3 = sns.distplot(df[df.PadID==500]['Vol'], fit=norm, kde=False, bins=20, vertical=True, color='#3ec727')  
	plt.ylabel('') 
	plt.xticks([]) 
	plt.yticks([]) 
	plt.savefig('%s/vol.pdf'%(args.data), bbox_inches='tight') 
	
	fig = plt.figure(figsize=(7,2)) 
	plt.subplots_adjust(wspace=0.1) 
	sns.set(font_scale=0.8) 
	gs = gridspec.GridSpec(1,2,width_ratios=[4,1]) 
	plt.subplot(gs[0]) 
	raw_line_plot1 = sns.lineplot(x='Timestep', y='Vol_norm', 
		data=df[(df.PadID==100) | (df.PadID==300) | (df.PadID==500)], 
		hue='PadID', style='PadID', palette=['b','r','g'], legend=False) 

	plt.xlabel('Timestep') 
	plt.ylabel('Normalized volume') 

	plt.subplot(gs[1]) 
	sns.set(font_scale=0.8) 

	raw_dist_plot1 = sns.distplot(df[df.PadID==100]['Vol_norm'], fit=norm, kde=False, bins=20, vertical=True, color='#0934c9') 
	raw_dist_plot2 = sns.distplot(df[df.PadID==200]['Vol_norm'], fit=norm, kde=False, bins=20, vertical=True, color='#ff0000') 
	raw_dist_plot3 = sns.distplot(df[df.PadID==300]['Vol_norm'], fit=norm, kde=False, bins=20, vertical=True, color='#3ec727')  
	plt.ylabel('') 
	plt.xticks([]) 
	plt.yticks([]) 
	plt.savefig('%s/norm_vol.pdf'%(args.data), bbox_inches='tight') 
	
	   	
	fig = plt.figure(figsize=(7,2)) 
	sns.set(font_scale=0.8) 
	sns.lineplot(x='Timestep', y='Vol_norm', data=df, ci='sd') 
	plt.xlabel('Timestep') 
	plt.ylabel('Normalized volume') 

	plt.savefig('%s/norm_all_vol.pdf'%(args.data), bbox_inches='tight') 
	
#	lags = autocorr1(pcb_mean, list(range(50))) 
#	print(lags) 
	fig = plt.figure() 
	plt.acorr(pcb_mean**2, maxlags=50) 
	plt.savefig('%s/auto.png'%(args.data)) 
		
	fig = plt.figure(figsize=(4,2.5)) 
	sns.set(font_scale=0.8) 
	sns.lineplot(x='Age', y='Vol_norm', data=df, ci='sd') 
	plt.xlabel('Timestep') 
	plt.ylabel('Normalized volume') 

	plt.savefig('%s/norm_norm_vol.pdf'%(args.data)) 
   
	fig = plt.figure() 
	sns.set_palette('Paired') 
	sns_scatter_plot = sns.scatterplot(x='X', y='Y', data=df, hue='Group', edgecolor='none', s=5, legend=False, palette=sns.color_palette(n_colors=len(shape_type))) 
	
	sns_scatter_plot.set(xticklabels=[], yticklabels=[]) 
	plt.savefig('%s/group.png'%(args.data)) 
	
	mean_norm = [] 
	std_norm = [] 
	
	for i in range(cycle): 
		mean_norm.append(df[df.Age==i]['Vol_norm'].mean())
		std_norm.append(df[df.Age==i]['Vol_norm'].std())
	
	pcb_info = {'mean': mean_norm, 
				'std': std_norm, 
				'height': height, 
				'width': width, 
				'pcb_mean': pcb_mean, 
				'pcb_std': pcb_std
				} 

	torch.save(pcb_info, '%s/pcb_info.pt'%(args.data)) 

#	df_info.to_csv('%s/df_info.csv'%(args.data_name), index=False)
	df.to_csv('%s/df_info.csv'%(args.data), index=False)

	test_pcbs = torch.zeros(all_timestep//cycle, cycle, 1, height, width) 

	# make testset
	for i in range(all_timestep): 

		df_time = df[df.Timestep==i] 
		x_ = df_time['X']
		y_ = df_time['Y'] 

		for x,y in zip(x_,y_):

			v = mean_norm[i%cycle] + std_norm[i%cycle]*torch.randn(1)[0].item()
			test_pcbs[i//cycle, i%cycle, 0, int(y), int(x)] = v

	torch.save(test_pcbs, '%s/test_pcbs.pt'%(args.data)) 

