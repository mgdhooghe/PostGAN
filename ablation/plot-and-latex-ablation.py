import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from random import sample

### ORGANIZE DATA ###
def organize_data(directory, file, ablation):
	print(directory+'/'+file)
	data = pd.read_csv(directory+'/'+file, index_col=0) #-1)
	data.fillna(0, inplace=True)

	####################
	#Organize data 
	####################
	dataset_name = ''
	protected = ''
	og_name = ''
	if 'adult' in directory:
	    dataset_name = 'Adult'
	    og_name = 'ADULT-SPLIT-TRAIN-'
	    protected = 'gender'
	elif 'compas' in directory:
	    dataset_name = 'Compas'
	    og_name = 'PROPUBLICA-COMPAS-SPLIT-TRAIN-'
	    if 'gender' in directory:
	        protected = 'gender'
	    else:
	        protected = 'race'
	    protected = 'race'
	elif 'german' in directory:
	    dataset_name = 'German'
	    og_name = 'GERMAN-SPLIT-TRAIN-'
	    protected = 'gender'
	elif 'bank' in directory:
	    dataset_name = 'Bank'
	    og_name = '59'
	    protected = 'age'
	elif 'medical' in directory:
	    dataset_name = 'Medical'
	    og_name = '20'
	    protected = 'race'

	## GAN DATA
	if dataset_name != 'Compas':
		select_folder = "mode_collapse/"+dataset_name.lower()+"_SELECT_FROM/selection-metrics-averages.csv"
	else:
		select_folder = "mode_collapse/compas-race_SELECT_FROM/selection-metrics-averages.csv"
	select_from = pd.read_csv(select_folder, index_col=0)
	select_from = select_from["VGAN-norm"]	
	select_from.name = "gan"

	## RENAME BASED ON ABLATION
	data.rename(index={og_name:'OG'},inplace=True)
	data.index = [ 'GAN' if re.sub('_\d+_\d+_\d+.csv','',txt) == 'sample_data' else txt for txt in data.index ]
	data.index = [str(txt).lower().replace('-','_').replace('sample_data_parallel_','').replace('sample_data_','').replace('meps_','').replace('_split','').replace(dataset_name.lower(),'').replace('_'+protected+'_','').replace('_'+protected,'').replace('_age','').replace('race','').replace('gender','').replace('sex','').replace('pro','').replace('_','',1).replace('_','',-1) for txt in data.index]
	if 'gan_type' not in ablation.lower():
		data.index = [str(txt).lower().replace('vgannorm','').replace('_','',1) for txt in data.index]
		data.index = [ 'base' if txt == 'vgan_norm' else txt for txt in data.index ]
		data.index = [ 'base' if txt == '' else txt for txt in data.index ]
	if 'FAIR_SWAP' in ablation or 'ALTERNATIVES' in ablation:
		data.index = [ 'US' if txt == 'base' else txt for txt in data.index ]
		data.index = [ 'FAIR_SWAP_BEFORE_EVEN' if 'nofair' in txt else txt for txt in data.index ]
		if 'FAIR_SWAP_BEFORE_EVEN' in data.index.values:	
			data = data.drop(index='FAIR_SWAP_BEFORE_EVEN')
		data.index = [ 'FAIR_SWAP' if txt == 'vgannorm' else txt for txt in data.index ]
	data.index = [ 'base' if txt == '' else txt for txt in data.index ]
	if 'select_percent' in ablation: 
		data.index = [ '1' if txt == 'base' else txt for txt in data.index ]
		data.index = [ str(int(100*float(txt))) if txt[0].isdigit() or txt[-1].isdigit() else txt for txt in data.index ]
	if 'set_percent' in ablation: 
		data.index = [ '5' if txt == 'base' else txt for txt in data.index ]
	if 'min_acc_bound' in ablation: 
		data.index = [ '0' if txt == 'base' else txt for txt in data.index ]
	if 'beta' in ablation: 
		data.index = [ '10' if txt == 'base' else txt for txt in data.index ]
	if 'average' in file:
		data = data.groupby(level=-1).mean()
	if 'avail_num' in ablation:
		data = data[data.index != 'base']
		data = data[data.index != 'Base']
	if 'no_test_split' in ablation:
		data.index = [ '1' if txt == 'base' else txt for txt in data.index ]

	return data, dataset_name, og_name, select_from, protected 

### PLOT DATA ###
def plot_data(data_folders, data_files, acc_measure, ablation):
	start = True 
	for directory in data_folders:
		if ablation != 'ALTERNATIVES':
			data_files = ["metrics-averages.csv"]
		fig, ax = plt.subplots(len(data_files) ,1, sharex=True, sharey=True)
		if len(data_files) > 1:
			ax[len(data_files)-1].set_xlabel('Disparate Impact')
		else:
			ax.set_xlabel('Disparate Impact')
		i = 0
		for file in data_files:
			if len(data_files) > 1:
				ax_alias = ax[i]
			else:
				ax_alias = ax
			if acc_measure == 'bal_acc':
				ax_alias.set_ylabel('Balaced Accuracy')
			else:
				ax_alias.set_ylabel('Accuracy')
				
			data, dataset_name, og_name, gan_data, protected = organize_data(directory, file, ablation)

			####### PLOT SAVED AS ########
			fig_name = 'mode_collapse/'+ablation+'_'+dataset_name+'_'+protected+'_'+acc_measure+'_plot.png'
			
			#########################
			# Plot Average Metrics
			#########################
			
			data['color'] = 'C0'
			c_i = 1
			for uind in sorted(data.index.unique()):
				data.loc[uind,'color'] = 'C'+str(c_i)
				c_i += 1
			
			## PLOT ERROR BARS
			og_data= data[data.index == 'original']
			#gan_data= data[data.index == 'gan']
			data= data[data.index != 'original']
			if not 'FAIR_SWAP' in ablation and not 'GAN_TYPE' in ablation and not 'TAB_FAIRGAN' in ablation and not 'ALTERNATIVES' in ablation:
				#data= data[data.index != 'base']
				#data.index = [ float(eval(i)) for i in data.index ]
				data = data.sort_index(axis=0)
                
			if 'beta' in ablation:
				data= data[data.index != 'gan']
				data.index = [ float(eval(i)) for i in data.index ]
				keep = [ ind < 1000 or ind >= float(10**10) for ind in data.index ]
				data = data[keep]
			if "gan" not in data.index.values:
				data = pd.concat([data,gan_data.to_frame().T])	
			data.index = [str(txt) for txt in data.index]
			c_i = 1
			ax_alias.scatter(og_data['DI'][0], og_data[acc_measure][0], color='k')
			for uind in sorted(data.index.unique()):
				if 'DI_sem' in data.columns:
					scatter_plot = ax_alias.errorbar(data.loc[uind,'DI'],data.loc[uind,acc_measure], xerr=data.loc[uind,'DI_sem'], yerr=data.loc[uind, acc_measure+'_sem'], fmt='o', ecolor='C'+str(c_i),c='none')
				if np.count_nonzero(data.index.values==uind) > 5:
					plot_sample = data.loc[uind].sample(5)
				else:
					plot_sample = data.loc[uind]
				scatter_plot = ax_alias.scatter(plot_sample['DI'],plot_sample[acc_measure],c='C'+str(c_i))
				c_i += 1
			box = ax_alias.get_position()
			ax_alias.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
			try:
				if len(data_files) > 1:
					plt.legend(['Original']+sorted(data.index.unique()), loc='lower center', ncol=6, bbox_to_anchor=(0.5,-0.5))
				else:
					plt.legend(['Original']+sorted(data.index.unique()), loc='lower center', ncol=6, bbox_to_anchor=(0.5,-0.25))
			except Exception as e:
				print(e)
			ax_alias.axhline(y=og_data[acc_measure][0],color='k', linestyle='dashed')
			ax_alias.axvline(x=og_data['DI'][0], color='k', linestyle='dashed')
			ax_alias.annotate('Og', (og_data['DI'][0], og_data[acc_measure][0]))
			## ANNOTATE POINTS
			if 'average' in file:
				for k, txt in enumerate(data.index):	
					txt = str(txt).lower().replace('sample_data_parallel_','').replace(dataset_name.lower(),'').replace('-'+protected,'').replace('_'+protected+'_','').replace('_sex_','').replace('-vgan-norm','')
					if txt == '':
						txt = 'Base'
					ax_alias.annotate(str(txt).lower(), (data['DI'].iloc[k], data[acc_measure].iloc[k]))
			## NEXT SUBPLOT
			i = i + 1

			## SAVE DATA
			if 'original' not in [ x for x in data.index ]:
				data = pd.concat([data,og_data])
			data['dataset'] = dataset_name
			if start and file == "metrics-averages.csv":
				collected_data = data
				start = False
			elif file == "metrics-averages.csv":
				collected_data = pd.concat([collected_data, data])
		## SAVE PLOT
		plt.suptitle(dataset_name+'-'+protected)
		fig.savefig(fig_name)
		## CLEAR PLOT
		plt.close()
		plt.clf()

	return collected_data

########################### LATEX ############################
def paper_latx(latx_data, latx_label, f, write, last=False):
	## BOLD BEST VALUES
	def max_bold(data, props=''):
		best = 'min'
		int_data = data.copy(deep=True)
		int_data.loc['Dataset'] = str(1)
		if 'acc' in latx_label:
			## BOLD ONLY MAX IGNORE ORIGINAL
			int_data['Original']=str(0)
			best = 'max'
			int_data.loc['Dataset'] = str(0)
			int_data = int_data.replace('nan',str(0))
		else:
			int_data = int_data.replace('nan',str(1))
		## FIND BEST
		#try:
		#	is_best = int_data.grouby(level=0).transform(best) == data
		#except:
		if True:
			if best == 'max':
				is_best = np.max(int_data) == data
			else:
				is_best = np.min(int_data) == data
		return np.where(is_best, 'bfseries:', '')

	## WRITE TO FILE 
	latx_data = latx_data.reset_index()
	latx_data.columns = [ x.replace('_','-') for x in latx_data.columns ]
	with open(f,write) as tf:
		if write == 'w':
			tf.write('\\begin{table*}\n')
			tf.write('\label{table: '+latx_label+'}\n')
			cap = f.split('/')[-1].split('.')[0]
			tf.write('\caption{'+cap+'}\n')
			tf.write('\\centering\n')
			#tf.write('\\tiny\n')
		s=latx_data.astype(str).style.hide_index()
		s=s.apply(max_bold, axis=1)
		#tf.write(s.to_latex(multirow_align="t",hrules=True))
		tf.write(s.to_latex(hrules=True))
		if last: 
			tf.write('\\end{table*}')
			tf.write('\\normalsize')
	return

#######MAIN#################
ablation = sys.argv[1]

##########################
#Read in data from select
##########################
directory = 'mode_collapse'
data_files = ["metrics-averages.csv", "metrics-plot-all.csv" ]
#acc_measures = ["bal_acc", "acc"]
acc_measures = ["bal_acc"]

## GET ALL DATA
data_folders = [ directory+'/'+folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory,folder)) ]
data_folders = [folder for folder in data_folders if folder.endswith(ablation)]
data_folders.sort()
print(data_folders)

data_dict = {}
##############
## PLOT DATA
##############
for acc_measure in acc_measures:
	data_dict[acc_measure] = plot_data(data_folders, data_files, acc_measure, ablation)


## ROUND ALL VALUES
def round_float(value):
	if isinstance(value, float):
		return round(value,3)
	else:
		return value


data_dict['bal_acc'] = data_dict['bal_acc'].applymap(round_float)
for metric in ['DI', 'bal_acc', 'acc', 'Stat Par', 'Eq Opp', 'Avg Odds', 'Theil']:
	## GET AVERAGE DATA
	di_data = data_dict['bal_acc'][['dataset',metric,metric+'_sem']]
	## COMBINE STANDARD ERROR DATA AND AVERAGE DATA
	di_data = di_data.applymap(str)
	di_data[metric] = di_data[metric].str.cat(di_data[metric+'_sem'], sep=" $\pm$ ")
	di_data = di_data.drop(columns=[metric+'_sem'])
	di_data = di_data.reset_index().pivot(index='dataset', columns='index',values=metric)
	di_data.columns = [ '\sys' if col == "US" else col for col in di_data.columns ]
	if ablation == 'ALTERNATIVES':
		if metric in ['DI','bal_acc','acc']:
			di_data = di_data[[ 'original', 'gan', 'fairgan', 'tab', 'fairswap', '\sys' ]]
		else:
			#di_data = di_data[[ 'original', 'gan', '\sys' ]]
			di_data = di_data[[ 'original', 'gan', 'fairgan', 'tab', 'fairswap', '\sys' ]]
	else:
		di_data.columns = [ float(str(x)) if str(x)[0].isdigit() or str(x)[0] == '.' else x for x in di_data.columns ]
		not_og = sorted([ x for x in di_data.columns if x != 'original' and x != 'gan' ])
		if ablation != 'beta':
			di_data = di_data[ [ 'original' ] + not_og ]
		else:
			di_data = di_data[ [ 'original', 'gan' ] + not_og ]
	di_data.columns = [ str(x).title() if x != '\sys' else str(x) for x in di_data.columns]
	di_data.index.names = ['Dataset']

	print(metric)
	print(di_data)

	##################
	## GET LATX FORMAT
	##################
	if ablation != 'ALTERNATIVES':
		f = directory+'/'+ablation+'_'+metric+'_latx.tex'
	elif metric == 'acc':
		f = directory+'/Accuracy.tex'
	elif metric == 'bal_acc':
		f = directory+'/AccuracyBal.tex'
	elif metric == 'DI':
		f = directory+'/DisparateImpact.tex'
	elif metric == 'Avg Odds':
		f = directory+'/AvgOdds.tex'
	elif metric == 'Stat Par':
		f = directory+'/StatisticalPar.tex'
	elif metric == 'Eq Opp':
		f = directory+'/EqualOpp.tex'
	elif metric == 'Theil':
		f = directory+'/Theil.tex'
	
	paper_latx(di_data, ablation+"-"+metric,f,'w',True)
	
