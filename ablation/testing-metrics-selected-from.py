
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


def run(file, og_data, og_file, test_data, protected, privileged, predicted, preferred, all=False):
	global og_dist

	if not all:
		#samples = get_data_plain(file, file)
		#og_data = get_data_plain(og_file, og_file)
		samples,_,_ = get_data(file, og_file, protected, privileged, predicted, preferred)
		samples = fill_x(og_data, samples)
	else:
		samples = file
		file = "All_files.csv"

	DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc, og_predicted_data = calc_metrics(samples, test_data, protected, predicted)
	DI = DI.round(3)
	prec = prec.round(3)
	rec = rec.round(3)
	acc = acc.round(3)
	bal_acc = bal_acc.round(3)
	print("Disp Imp: ",DI, " Precision: ",prec," Recall: ",rec," Accuracy: ",acc, " Bal Acc: ", bal_acc)
	if file == og_file:
		file = 'Original'

	##GET DIVERSITY OF DATASET
	from sklearn.decomposition import PCA
	
	# Assuming 'data' is your dataset
	pca = PCA(n_components=1)
	pca.fit(samples)
	pca_data = pca.fit_transform(samples)
	print(pca_data)
	
	# Explained variance ratio gives the proportion of variance explained by each component
	explained_variance_ratio = pca.explained_variance_ratio_
	print("Explained variance ratio by each principal component:", explained_variance_ratio)
	print(sum(explained_variance_ratio))
	variances = np.var(pca_data, axis=0)



	return [file, DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc, variances[0]], samples

	
    
def get_og_file(input_file, protected):
    if 'adult' in input_file:
        og_file = 'adult/ADULT-SPLIT-TRAIN-60.csv'
        og_test = 'adult/ADULT-SPLIT-TEST-20.csv'
        VGAN_norm = '../adult/adult-VGAN-1hot-norm-split-60/9/'
        VGAN_gmm = '../adult/adult-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../adult/adult-VGAN-WTRAIN-split/'
        WGAN_gmm = '../adult/adult-VGAN-1hot-gmm-split-60-WTrain-best/8-9/' 
        TABFAIRGAN = '../adult/adult-TABFAIRGAN/'
        FAIRGAN = '../adult/adult-FAIRGAN/'
    elif 'compas' in input_file:
        og_test = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TEST-20.csv'
        og_file = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TRAIN-60.csv'
        VGAN_norm = '../compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/' 
        VGAN_gmm = '../compas/compas-pro-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../compas/compas-pro-VGAN-WTRAIN-split/'
        WGAN_gmm = '../compas/compas-pro-VGAN-1hot-gmm-split-60-WTrain/8-9/'
        TABFAIRGAN = '../compas/compas-pro-TABFAIRGAN/' + protected + '/'
        FAIRGAN = '../compas/compas-pro-FAIRGAN/' + protected + '/'
    elif 'german' in input_file:
        og_file = 'german/GERMAN-SPLIT-TRAIN-60.csv'
        og_test = 'german/GERMAN-SPLIT-TEST-20.csv'
        VGAN_norm = '../german/german-VGAN-1hot-split-validation-best/9/'
        VGAN_gmm = '../german/german-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../german/german-VGAN-WTRAIN-split/'
        WGAN_gmm = '../german/german-VGAN-1hot-gmm-split-60-WTrain/8-9/'
        TABFAIRGAN = '../german/german-TABFAIRGAN/'
        FAIRGAN = '../german/german-FAIRGAN/'
    elif 'bank' in input_file:
        og_file = 'bank/bank-full-SPLIT-TRAIN-60-age.csv'
        og_test = 'bank/bank-full-SPLIT-TEST-20-age.csv'
        VGAN_norm = '../bank/bank-VGAN-best/9-age/'
        VGAN_gmm = '../bank/bank-full-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../bank/bank-VGAN-WTRAIN-split/'
        WGAN_gmm = '../bank/bank-full-VGAN-1hot-gmm-split-60-WTrain/8-9/'
        TABFAIRGAN = '../bank/bank-TABFAIRGAN/'
        FAIRGAN = '../bank/bank-FAIRGAN/'
    elif 'medical' in input_file:
        og_file = 'medical/meps21-SPLIT-TRAIN-60.csv'
        og_test = 'medical/meps21-SPLIT-TEST-20.csv'
        VGAN_norm = '../medical/medical-VGAN-1hot-norm-split-60/8-9/'
        VGAN_gmm = '../medical/medical-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../medical/medical-WTRAIN-VGAN-1hot-norm-split-60/8-9/'
        WGAN_gmm = '../medical/medical-WTRAIN-VGAN-1hot-gmm-split-60-best/8-9/'
        TABFAIRGAN = '../medical/medical-TABFAIRGAN/'
        FAIRGAN = '../medical/medical-FAIRGAN/'
    return og_file, og_test, VGAN_norm, VGAN_gmm, WGAN_norm, WGAN_gmm, TABFAIRGAN, FAIRGAN

if __name__ == "__main__":
	
	mets = True 
	dataset = sys.argv[1]
	protected = sys.argv[2]
	if len(sys.argv) > 3:
		mets = False
		mets_avg = pd.read_csv(sys.argv[3])
		mets_avg = mets_avg.T
		mets_avg.columns = mets_avg.iloc[0]
		mets_avg = mets_avg.iloc[1:]
		print(mets_avg)
    
	dataset_name = 'unknown'
	possible_datasets = ['adult','bank','compas','german','medical']
	for data_name in possible_datasets:
		if data_name in dataset.lower():
			dataset_name = data_name
			break

	og_file, test_file, VGAN_norm, VGAN_gmm, WGAN_norm, WGAN_gmm, TABFAIRGAN, FAIRGAN = get_og_file(dataset_name, protected)
		
	print("Original: ",og_file," Test: ",test_file)

	VGAN_norm = [ VGAN_norm+'/'+file for file in os.listdir(VGAN_norm) if file.startswith('sample_data') ] 
	VGAN_gmm = [ VGAN_gmm+'/'+file for file in os.listdir(VGAN_gmm) if file.startswith('sample_data') ] 
	WGAN_norm = [ WGAN_norm+'/'+file for file in os.listdir(WGAN_norm) if file.startswith('sample_data') ] 
	WGAN_gmm = [ WGAN_gmm+'/'+file for file in os.listdir(WGAN_gmm) if file.startswith('sample_data') ] 
	FAIRGAN = [ FAIRGAN +'/'+file for file in os.listdir(FAIRGAN) if file.startswith('sample_data') ] 
	TABFAIRGAN = [ TABFAIRGAN +'/'+file for file in os.listdir(TABFAIRGAN) if file.startswith('sample_data') ] 

	data_files = []
	for folder in [VGAN_norm, VGAN_gmm, WGAN_norm, WGAN_gmm, TABFAIRGAN, FAIRGAN]:
		data_files = data_files + folder 

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

	def get_var(samples):
		##GET DIVERSITY OF DATASET
		from sklearn.decomposition import PCA
		
		# Assuming 'data' is your dataset
		pca = PCA(n_components=1)
		pca.fit(samples)
		pca_data = pca.fit_transform(samples)
		variances = np.var(pca_data, axis=0)
		return variances[0]

	metric_columns=['file','DI', 'Stat Par', 'Avg Odds','Eq Opp','Theil','prec','rec','acc', 'bal_acc', 'var']
	if mets:
		try:
			mets_pd = pd.read_csv('mode_collapse/'+dataset+'/selection-metrics.csv',index_col=0)
			print(mets_pd)
		except:
			mets_pd = pd.DataFrame(columns=metric_columns)
			print("NEW DATAFRAME")
		for data_file in data_files:
			print(data_file)
			data_file_test = data_file
			if data_file_test == og_file:
				data_file_test = "Original"
			print(data_file_test)
			done_files = [file for file in mets_pd['file']]
			if data_file_test not in done_files:
				print("RUNNING FILE")
				m, data = run(data_file, og_data, og_file, test_data, protected, privileged, predicted, preferred)
				mets_pd.loc[len(mets_pd.index)] = m
				mets_pd.to_csv('mode_collapse/'+dataset+'/selection-metrics.csv')
				 
		print(mets_pd)
		mets_pd.to_csv('mode_collapse/'+dataset+'/selection-metrics.csv')


	
		mets_pd['config'] = mets_pd['file']
		mets_pd['config'] = ['VGAN-norm' if x in VGAN_norm else x for x in mets_pd['config']]
		mets_pd['config'] = ['VGAN-gmm' if x in VGAN_gmm else x for x in mets_pd['config']]
		mets_pd['config'] = ['WGAN-norm' if x in WGAN_norm else x for x in mets_pd['config']]
		mets_pd['config'] = ['WGAN-gmm' if x in WGAN_gmm else x for x in mets_pd['config']]
		mets_pd['config'] = ['FAIRGAN' if x in FAIRGAN else x for x in mets_pd['config']]
		mets_pd['config'] = ['TABFAIRGAN' if x in TABFAIRGAN else x for x in mets_pd['config']]

		for col in mets_pd.columns:
			if isinstance(mets_pd[col][0],float):
				mets_pd[col] = mets_pd[col].abs()
		mets_avg = mets_pd.groupby(['config']).mean()
		print(mets_avg)
	mets_avg['all var'] = 0

	file_set = ['VGAN-norm', 'VGAN-gmm', 'WGAN-norm', 'WGAN-gmm', 'TABFAIRGAN', 'FAIRGAN']
	rounds = 0
	print(mets_avg)
	for folder in [VGAN_norm, VGAN_gmm, WGAN_norm, WGAN_gmm, TABFAIRGAN, FAIRGAN]:
		first = True
		gan_type = file_set[rounds]
		for file in folder:
			samples,_,_ = get_data(file, og_file, protected, privileged, predicted, preferred)
			samples = fill_x(og_data, samples)
			if first:
				all_samples = samples
				first = False
			else:
				all_samples = pd.concat([all_samples, samples])
		
		all_var = get_var(all_samples)	
		print('VAR: ',all_var)	
		print('GAN TYPE: ', gan_type)
		print(mets_avg['all var'])
		mets_avg.loc[gan_type,'all var'] = all_var
		print(mets_avg.loc[gan_type, 'all var'])
		rounds = rounds+1

	if mets:
		mets_std_err = mets_pd.groupby(['config']).sem()
		mets_std_err.columns = [ col+'_sem' for col in mets_std_err.columns ]
		mets_avg = pd.concat([mets_avg, mets_std_err], axis=1)
		mets_avg.sort_index(inplace=True)
	mets_avg = mets_avg.T
	mets_avg.to_csv('mode_collapse/'+dataset+'/selection-metrics-averages.csv')
