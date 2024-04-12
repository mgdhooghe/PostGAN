
import warnings
import os
import sys
import re
warnings.filterwarnings("ignore")
#sys.stderr = open(os.devnull, 'w')

from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance
from sklearn.decomposition import PCA

# Turn on scikit-learn optimizations :
from sklearnex import patch_sklearn
patch_sklearn()

import aif360.datasets as datasets
import aif360.metrics as metrics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from transform_main import *
from var_utils import *

def fill_x(og_data, data):
        diff = list(set(og_data.columns)-set(data.columns))
        if len(diff) == 0:
                return data
        add_cols = pd.DataFrame(0, index=np.arange(len(data)), columns=diff)
        data= pd.concat([data,add_cols],axis=1)
        return data[list(og_data.columns)]

def calc_min_change_metrics(data, protected, predicted):
	privileged = 1
	unprivileged = 0
	preferred = 1
	unpreferred = 0
	priv_groups = [{protected: privileged}]
	unpriv_groups = [{protected: unprivileged}]
	if 'fnlwgt' in data.columns:
		weights_name = 'fnlwgt'
	else:
		weights_name = None

	## PERFECT ACCURACY
	dataset= datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
	no_pred_classMetrics = metrics.ClassificationMetric(dataset, dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)
	nopred_DI =  1-min(no_pred_classMetrics.disparate_impact(), 1/(no_pred_classMetrics.disparate_impact()))
	print("IF PERFECT ACC - DISP= ", nopred_DI)
	## VALUE_COUNTS
	val_counts = data.value_counts([protected, predicted])
	print(val_counts)
	## PROBABILITIES
	s0 = data[data[protected] == 0]
	y1_given_s0 = s0[s0[predicted] == 1]
	prob_y1_given_s0 = len(y1_given_s0)/len(s0)
	s1 = data[data[protected] == 1]
	y1_given_s1 = s1[s1[predicted] == 1]
	prob_y1_given_s1 = len(y1_given_s1)/len(s1)
	y0 = len(data[data[predicted] == 0].index)
	y1 = len(data[data[predicted] == 1].index)
	print("s0: ", len(s0.index))
	print("s1: ", len(s1.index))
	print("y1|s0: ",len(y1_given_s0.index))
	print("y1|s1: ",len(y1_given_s1.index))
	print("prob(y1|s0): ",prob_y1_given_s0)
	print("prob(y1|s1): ",prob_y1_given_s1)
	## MINIMUM CHANGE
	s0_len = len(s0.index)
	s1_len = len(s1.index)
	y1_given_s0_len = len(y1_given_s0.index)
	y1_given_s1_len = len(y1_given_s1.index)
	A = y1_given_s0_len
	B = s0_len
	C = y1_given_s1_len
	D = s1_len	
	################ CHANGES TO GET PERFECT DI #####################
	print("################### DI = 1 #################")
	## IF ONLY CHANGING UNDERPRIVILEGED
	i_u = int(B*C/D - A)
	print("ONLY CHANGING UNDERPRIVILEGED: ",i_u)
	print("NEW DISP: ", ((A+i_u)/B)/(C/D))
	print("NEW ACC: ", (B+D-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*( y1/(y1+max(i_u,0))+ y0/(y0+max(-i_u,0))))
	## IF ONLY CHANGING PRIVILEGED
	i_p = int(A*D/B - C)
	print("ONLY CHANGING PRIVILEGED: ",i_p)
	print("NEW DISP: ", (A/B)/((C+i_p)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0))+ y0/(y0+max(i_p,0))))
	## IF CHANGING HALF OF EACH
	print("HALF CHANGING UNDERPRIVILEGED: ",i_u/2)
	print("HALF CHANGING PRIVILEGED: ",i_p/2)
	print("TOTAL CHANGED: ",np.abs(i_p/2)+np.abs(i_u/2))
	print("NEW DISP: ", ((A+i_u/2)/B)/((C+i_p/2)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p)-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0)+max(i_u,0))+ y0/(y0+max(i_p,0)+max(-i_u,0))))
	################ CHANGES TO GET LEGAN DI (.8) #####################
	print("################### DI = .8 #################")
	## IF ONLY CHANGING UNDERPRIVILEGED
	i_u = int(.8*B*C/D - A)
	print("ONLY CHANGING UNDERPRIVILEGED: ",i_u)
	print("NEW DISP: ", ((A+i_u)/B)/(C/D))
	print("NEW ACC: ", (B+D-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*( y1/(y1+max(i_u,0))+ y0/(y0+max(-i_u,0))))
	## IF ONLY CHANGING PRIVILEGED
	i_p = int((A*D/B - .8*C)/.8)
	print("ONLY CHANGING PRIVILEGED: ",i_p)
	print("NEW DISP: ", (A/B)/((C+i_p)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0))+ y0/(y0+max(i_p,0))))
	## IF CHANGING HALF OF EACH
	print("HALF CHANGING UNDERPRIVILEGED: ",i_u/2)
	print("HALF CHANGING PRIVILEGED: ",i_p/2)
	print("TOTAL CHANGED: ",np.abs(i_p/2)+np.abs(i_u/2))
	print("NEW DISP: ", ((A+i_u/2)/B)/((C+i_p/2)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p)-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0)+max(i_u,0))+ y0/(y0+max(i_p,0)+max(-i_u,0))))
	
	return

		


def calc_metrics(data, og_test_data, protected, predicted):
	alg = 'RandomForest'

	privileged = 1
	unprivileged = 0
	preferred = 1
	unpreferred = 0
	priv_groups = [{protected: privileged}]
	unpriv_groups = [{protected: unprivileged}]

	if 'fnlwgt' in og_test_data.columns:
		weights_name = 'fnlwgt'
	else:
		weights_name = None
	DI = 0
	stat_par = 0
	avg_odds = 0
	eq_op = 0
	theil = 0 

	bal_acc = 0
	prec = 0 
	rec = 0
	acc = 0
	avg = 3 
	if protected == 'gender' and predicted == 'labels':
		avg = 6
	for i in range(avg):
		###PREDICT OG DATA
		gen_reg = train_on_data(data)
		og_predicted_data = predict_output(og_test_data, gen_reg) #Predict original on generated trained classifier

		## ORGANIZE ORIGINAL DATA
		og_dataset= datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=og_test_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
		###ORGANIZE DATA TRAINED ON GENERATED AND TESTED ON ORIGINAL TEST DATA
		og_predicted_dataset = datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=og_predicted_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
		og_classMetrics = metrics.ClassificationMetric(og_dataset, og_predicted_dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)# Predicted Original data on generated trained classifer
		
		## DISPARATE IMPACE
		DI_intermede =  1-min(og_classMetrics.disparate_impact(), 1/(og_classMetrics.disparate_impact()))
		print(DI_intermede) 
		DI = DI + DI_intermede
		## STATISTICAL PARITY
		stat_par = stat_par + og_classMetrics.statistical_parity_difference()
		## AVERAGE ODDS
		avg_odds = avg_odds + og_classMetrics.average_odds_difference()
		## EQUAL OPPORTUNITY
		eq_op = eq_op + og_classMetrics.equal_opportunity_difference()
		## THIEL
		theil = theil + og_classMetrics.theil_index()
		## BALANCED ACCURACY
		bal_acc = bal_acc + 0.5*(og_classMetrics.true_positive_rate()+og_classMetrics.true_negative_rate())
		## PRECISION
		prec =  prec + og_classMetrics.precision()
		## RECALL
		rec = rec + og_classMetrics.recall()
		## ACCURACY
		acc =  acc + og_classMetrics.accuracy()
	DI = DI/avg 
	avg_odds = avg_odds/avg
	eq_op = eq_op/avg
	theil = theil/avg
	bal_acc = bal_acc/avg 
	prec = prec/avg 
	rec = rec/avg 
	acc = acc/avg 
	return DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc, og_predicted_data


def run(file, og_data, og_file, test_data, protected, privileged, predicted, preferred, basic, all=False):
	global og_dist

	if not all:
		#samples = get_data_plain(file, file)
		#og_data = get_data_plain(og_file, og_file)
		samples,_,_ = get_data(file, og_file, protected, privileged, predicted, preferred)
		samples = fill_x(og_data, samples)
	else:
		samples = file
		file = "All_files.csv"

	DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc, predicted_data = calc_metrics(samples, test_data, protected, predicted)
	DI = DI.round(3)
	stat_par = stat_par.round(3)
	avg_odds = avg_odds.round(3)
	eq_op = eq_op.round(3)
	theil = theil.round(3)
	prec = prec.round(3)
	rec = rec.round(3)
	acc = acc.round(3)
	bal_acc = bal_acc.round(3)
	print("Disp Imp: ",DI, " Precision: ",prec," Recall: ",rec," Accuracy: ",acc, " Bal Acc: ", bal_acc)
	if file == og_file:
		file = 'Original'
	if basic in file: 
		file = 'Base'

	##GET DIVERSITY OF DATASET
	from sklearn.decomposition import PCA

	# Assuming 'data' is your dataset
	pca = PCA(n_components=1)
	pca.fit(samples)
	pca_data = pca.fit_transform(samples)

	# Explained variance ratio gives the proportion of variance explained by each component
	explained_variance_ratio = pca.explained_variance_ratio_
	#print("Explained variance ratio by each principal component:", explained_variance_ratio)
	#print(sum(explained_variance_ratio))
	variances = np.var(pca_data, axis=0)
	#print("Variances for each feature:", variances)

	return [file.split('/')[-1], DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc], samples
	
    
def get_og_file(input_file, protected):
    if 'adult' in input_file:
        og_file = 'adult/ADULT-SPLIT-TRAIN-60.csv'
        og_test = 'adult/ADULT-SPLIT-TEST-20.csv'
        basic = 'ablation_study_GAN_type/adult-VGAN-norm_selected_samples/'
    elif 'compas' in input_file:
        og_test = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TEST-20.csv'
        og_file = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TRAIN-60.csv'
        basic = 'ablation_study_GAN_type/compas-'+protected+'-VGAN-norm_selected_samples/'
    elif 'german' in input_file:
        og_file = 'german/GERMAN-SPLIT-TRAIN-60.csv'
        og_test = 'german/GERMAN-SPLIT-TEST-20.csv'
        basic = 'ablation_study_GAN_type/german-VGAN-norm_selected_samples/'
    elif 'bank' in input_file:
        og_file = 'bank/bank-full-SPLIT-TRAIN-60-age.csv'
        og_test = 'bank/bank-full-SPLIT-TEST-20-age.csv'
        basic = 'ablation_study_GAN_type/bank-VGAN-norm_selected_samples/'
    elif 'medical' in input_file:
        og_file = 'medical/meps21-SPLIT-TRAIN-60.csv'
        og_test = 'medical/meps21-SPLIT-TEST-20.csv'
        basic = 'ablation_study_GAN_type/medical-VGAN-norm_selected_samples/'
    return og_file, og_test, basic

if __name__ == "__main__":
	
	dataset = sys.argv[1]
	protected = sys.argv[2]
	ablation_folder = sys.argv[3]
    
	dataset_name = 'unknown'
	basic = ''
	possible_datasets = ['adult','bank','compas','german','medical']
	for data_name in possible_datasets:
		if data_name in dataset.lower():
			dataset_name = data_name
			break
	og_file, test_file, basic = get_og_file(dataset_name, protected)
		
	print("Original: ",og_file," Test: ",test_file, " Basic: ",basic)
	data_folders = [ ablation_folder+'/'+folder for folder in os.listdir(ablation_folder) if os.path.isdir(os.path.join(ablation_folder,folder)) ] 
	data_folders = [ folder for folder in data_folders if dataset_name in folder ]
	data_files =  [ ablation_folder+'/'+file for file in os.listdir(ablation_folder) if file.startswith('sample_data') ] 
	data_files = [ folder for folder in data_files if dataset_name in folder ]
	for testing_folder in data_folders:
		data_files = data_files + [ testing_folder+'/'+file for file in os.listdir(testing_folder) if file.startswith('sample_data') ] 
	try:
		if 'disp_select_all' not in ablation_folder and 'GAN_type' not in ablation_folder: 
			data_files = data_files + [basic+'/'+file for file in os.listdir(basic) if file.startswith('sample_data')]
	except:
		print("NO BASIC")

	if protected == 'gender':
		data_files = [x for x in data_files if not 'race' in x]
	if protected == 'race':
		data_files = [x for x in data_files if not 'gender' in x]
		data_files = [x for x in data_files if not 'sex' in x]
	data_files = [og_file] + sorted(list(set(data_files)))
	print(data_files)
	protected, privileged, _, predicted, preferred, _ = get_labels(og_file, protected)
	og_data,_,_ = get_data(og_file, og_file, protected, privileged, predicted, preferred)
	test_data,_,_ = get_data(test_file, og_file, protected, privileged, predicted, preferred)
	test_data = fill_x(og_data, test_data)
	all_data = pd.DataFrame(columns=og_data.columns)
	#calc_min_change_metrics(test_data, protected, predicted)

	if not os.path.exists('mode_collapse/'+dataset+'/'):
		os.makedirs('mode_collapse/'+dataset+'/')

	metric_columns=['file','DI', 'Stat Par', 'Avg Odds','Eq Opp','Theil','prec','rec','acc', 'bal_acc']

	try:
		mets_pd = pd.read_csv('mode_collapse/'+dataset+'/metrics.csv',index_col=0)
		print(mets_pd)
		if list(mets_pd.columns) != metric_columns:
			mets_pd = pd.DataFrame(columns=metric_columns)
		print(mets_pd)
	except:
		mets_pd = pd.DataFrame(columns=metric_columns)
		print("NEW DATAFRAME")
	done_files = [file for file in mets_pd['file']]
	for data_file in data_files:
		print(data_file)
		file_name = data_file
		try:
			if file_name == og_file:
				file_name = 'Original' 
			elif file_name in [basic+'/'+file for file in os.listdir(basic) if file.startswith('sample_data')]:
				file_name = 'Base'
		except:
			print("NO BASIC")
		if file_name.split('/')[-1] not in done_files:
			print('RUNNING FILE')
			m, data = run(data_file, og_data, og_file, test_data, protected, privileged, predicted, preferred, basic)
			mets_pd.loc[len(mets_pd.index)] = m
			mets_pd.to_csv('mode_collapse/'+dataset+'/metrics.csv')
			print(mets_pd)
			mets_pd.to_csv('mode_collapse/'+dataset+'/metrics.csv')
	for col in mets_pd.columns:
		print(col)
		if col != "file":
			mets_pd[col] = mets_pd[col].abs()
	mets_pd['config'] = [ re.sub(r'[\d+_]+(-age)?.csv$', '', x) for x in mets_pd['file'] ] 
	mets_pd_plot = mets_pd.copy(deep=True)
	mets_pd_plot.index = [ re.search(r'\.?[\d,*]+',val).group() if re.search(r'\d+',val) else val for val in mets_pd_plot['config'] ]
	mets_pd_plot.to_csv('mode_collapse/'+dataset+'/metrics-plot-all.csv')
	mets_avg = mets_pd.groupby(['config']).mean()
	mets_avg.index = [ re.search(r'\.?[\d,*]+',val).group() if re.search(r'\d+',val) else val for val in mets_avg.index ]
	print(mets_avg)
	mets_std_err = mets_pd.groupby(['config']).sem()
	mets_std_err.columns = [ col+'_sem' for col in mets_std_err.columns ]
	mets_std_err.index = [ re.search(r'\.?[\d,*]+',val).group() if re.search(r'\d+',val) else val for val in mets_std_err.index ]
	mets_avg = pd.concat([mets_avg, mets_std_err], axis=1)
	mets_avg.sort_index(inplace=True)
	mets_avg.to_csv('mode_collapse/'+dataset+'/metrics-averages.csv')
