from transform_data_dataset import *
import math
import numpy as np
import pandas as pd
import torch
import tracemalloc
import gc
import time

import sys
import os
import aif360.metrics as metrics
import aif360.datasets as datasets

# Turn on scikit-learn optimizations :
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_train_test(data, split=True):
	og_x = data[data.columns[:-1]].astype(float)
	og_y = data[data.columns[-1]].astype(float)
	if not split:
		return og_x, og_y
	x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=0.33, stratify=og_y)
	return x_train, x_test, y_train, y_test

def train_on_data(data, return_test=False, return_train=False, split=True, new_alg=None):
	global alg
	if new_alg!=None:
		alg=new_alg

	if alg == 'LogReg':
		reg = LogisticRegression(max_iter=5000)
	elif alg == 'Percep':
		reg = Perceptron()
	elif alg == 'LinSVC':
		reg = LinearSVC(max_iter=5000)
	elif alg == 'Tree':
		reg = DecisionTreeClassifier() 
	elif alg == 'RandomForest':
		reg = RandomForestClassifier(n_estimators=500) 
	elif alg == 'NeuralNet':
		reg = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), max_iter=5000)
	
	if split:
		x_train, x_test, y_train, y_test = get_train_test(data, split)
	else:
		x_train, y_train = get_train_test(data, split)

	if 'fnlwgt' in data.columns:
		weight = x_train['fnlwgt']
		x_train_with_weight = x_train.copy(deep=True)
		x_train = x_train.drop(columns=['fnlwgt'])
	else:
		weight = None	

	try:
		if alg != 'NeuralNet':
			reg.fit(x_train,y_train, sample_weight = weight)
		else:
			reg.fit(x_train,y_train)
	except Exception as e:
		print("FAILED TRAINING: ",e)
		reg = -1

	if 'fnlwgt' in data.columns:
		x_train = x_train_with_weight
	if return_test:
		test = pd.concat([x_test,y_test],axis=1).reset_index(drop=True)
		if return_train:
			train = pd.concat([x_train,y_train],axis=1).reset_index(drop=True)
			return reg, test, train
		return reg, test 
	return reg

def predict_output(data, reg):
	x = data[data.columns[:-1]]
	x = pd.DataFrame(x).reset_index(drop=True)
	if 'fnlwgt' in x.columns:
		x_with_weight = x.copy(deep=True)
		x = x.drop(columns = ['fnlwgt'])
	if reg == -1:
		y_pred = np.zeros([len(data)])
	else:
		try:
			y_pred = reg.predict(x)
		except:
			y_pred = np.zeros([len(data)])
	if 'fnlwgt' in data.columns:
		x = x_with_weight
	return pd.concat([x,pd.Series(y_pred,name=data.columns[-1])], axis=1)

def fill_x(og_data, data):
	diff = list(set(og_data.columns)-set(data.columns))
	if len(diff) == 0:
		return data
	add_cols = pd.DataFrame(0, index=np.arange(len(data)), columns=diff) 
	data= pd.concat([data,add_cols],axis=1)
	return data[list(og_data.columns)]

def compute_kl(real, pred):
    return np.sum((np.log(pred + 1e-4) - np.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_real, ohe, protected='', predicted=''):
    # Get predicted
    kl = 0.0
    x_fake = x_fake.astype(np.float)
    x_real = x_real.astype(np.float)
    for i in range(ohe.n_features_in_):
        cat = ohe.categories_[i]
        name = ohe.feature_names_in_[i]
        #Ignore Protected and Predicted
        if name != protected and name != predicted:
            cat = [name+'_'+col for col in cat]
            fakex = np.array(x_fake[cat].values)
            realx = np.array(x_real[cat].values)
            dist = np.sum(fakex, axis=0)
            dist = dist / np.sum(dist)
                
            real = np.sum(realx, axis=0)
            real = real / np.sum(real)
                
            kl += compute_kl(real, dist)
    return kl
    

#Accuracy Loss
def ALoss(dataMetrics, ogdataMetrics, data, original_data, ohe, values="all", protected='', predicted=''):
	if values == "all":
		A1 = dataMetrics.accuracy() # 
		A2 = ogdataMetrics.accuracy() # 
		A3 = 0.5*(dataMetrics.true_positive_rate()+dataMetrics.true_negative_rate())
		A3B = 0.5*(ogdataMetrics.true_positive_rate()+ogdataMetrics.true_negative_rate())
		A4 = KL_Loss(data.convert_to_dataframe()[0], original_data.convert_to_dataframe()[0], ohe) 
		vals = [A1, A2, A3, A3B, A4]
		vals = [0 if math.isnan(x) else x for x in vals]
		print('A: ',vals)
		return vals, ['Accuracy Gen', 'Accuracy Orig', 'Accuracy Bal Gen', 'Accuracy Bal Orig','K1 Dist']
	if values == "bal":
		A3 = 0.5*(dataMetrics.true_positive_rate()+dataMetrics.true_negative_rate())
		return [A3], ['Accuracy Bal']
	if values == "KL":
		A4 = KL_Loss(data.convert_to_dataframe()[0], original_data.convert_to_dataframe()[0], ohe).item() 
		return [A4], ['Kl Dist']

#Fairness Loss	
def FLoss(data, values="fair", gen=None):
	try:
		if data.disparate_impact() == 0 or math.isnan(data.disparate_impact()):
			F1 = 1
		else:
			F1 = 1-min(data.disparate_impact(), 1/data.disparate_impact())
	except ZeroDivisionError:
		F1 = 1
	if values == "disp" or values == "disp/bal":
		return [F1], ['Disparate Impact']
	
	F2 =  data.equal_opportunity_difference()
	F3 =  data.statistical_parity_difference()
	F4 =  data.theil_index()
	F5 =  data.average_odds_difference() 

	if gen!=None:
		try:
			if gen.disparate_impact() == 0 or math.isnan(gen.disparate_impact()):
				F12 = 1
			else:
				F12 = 1-min(gen.disparate_impact(), 1/gen.disparate_impact())
		except ZeroDivisionError:
			F12 = 1
		vals = [F1, F12, F2, F3, F4, F5]
		print('F: ',vals)
		vals = [1 if math.isnan(x) else x for x in vals]
		return vals, ['Disparate Impact', 'Disparate Impact Gen', 'Equal Opp. Diff', 'Statistical Par. Diff', 'Theil Index','Avg. Odds']
	vals = [F1, F2, F3, F4, F5]
	print('F: ',vals)
	vals = [1 if math.isnan(x) else x for x in vals]
	return vals, ['Disparate Impact', 'Equal Opp. Diff', 'Statistical Par. Diff', 'Theil Index','Avg. Odds']


#Read all files	
def read_files(file, og_file, og_test_data, og_train_data, prot_var, ohe, og=False):

	if og==True:
		file_path = og_file
		data = og_train_data
	else:	
		file_path = dir + '/' + file
		data,_,_ = get_data(file_path, prot_var)
		if 'Unnamed: 0' in data.columns:
			data = data.drop(columns=['Unnamed: 0'])
		data = fill_x(og_train_data, data).dropna().reset_index(drop=True)
	aloss, aloss_labels, floss, floss_labels = calculate(data, og_file, og_test_data, ohe, prot_var) 
	return aloss, aloss_labels, floss, floss_labels 


def calculate(all_data, og_file, og_test_data, ohe, prot_var, values="all"):
	global alg
	print('ALG: ',alg)
	og_data = og_test_data
	## Vars
	protected, _, _, predicted, _, _ = get_labels(og_file, prot_var)
	privileged = 1
	unprivileged = 0
	preferred = 1
	unpreferred = 0
	priv_groups = [{protected: privileged}] 
	unpriv_groups = [{protected: unprivileged}] 
	## Setup Data
	if alg == 'NeuralNet':
 		get_avg = 3 
	else:
		get_avg = 5 

	if 'fnlwgt' in og_test_data.columns:
		weights_name = 'fnlwgt'
	else:
		weights_name = None
	## ORGANIZE ORIGINAL DATA
	og_dataset= datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=og_test_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name) 
	for i in range(get_avg):
		###PREDICT OG DATA
		gen_reg = train_on_data(all_data, split=False)
		og_predicted_data = predict_output(og_test_data, gen_reg) #Predict original on generated trained classifier

		###ORGANIZE DATA TRAINED ON GENERATED AND TESTED ON ORIGINAL TEST DATA
		og_predicted_dataset = datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=og_predicted_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
		og_classMetrics = metrics.ClassificationMetric(og_dataset, og_predicted_dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)# Predicted Original data on generated trained classifer	

		###PREDICT GENERATED DATA
		gen_reg, data = train_on_data(all_data, return_test=True, split=True)
		predicted_data = predict_output(data, gen_reg) #predict test data output

		###ORGANIZE DATA TRAINED ON GENERATED AND TESTED ON GENERATED	
		dataset = datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
		predicted_dataset = datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=predicted_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
		classMetrics = metrics.ClassificationMetric(dataset, predicted_dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)	
			
		###CALCULATE FAIRNESS
		counts = og_predicted_data[[protected,predicted]].value_counts()
		print(counts)
		counts = counts.reindex(((0,0),(0,1),(1,1),(1,0)),fill_value=0)	
		d_u = counts[(0,1)]/(counts[(0,1)]+counts[(0,0)])
		d_p = counts[(1,1)]/(counts[(1,1)]+counts[(1,0)])
		d = 1-min(d_u/d_p,d_p/d_u)
		print('CALC disp: ',d)
		
		this_floss, floss_labels = FLoss(og_classMetrics, values, gen=classMetrics)
		if i == 0:
			floss = np.array(this_floss)*0
		floss = floss + np.array(this_floss)
		###CALCULATE ACCURACY
		this_aloss, aloss_labels = ALoss(classMetrics, og_classMetrics, dataset, og_dataset, ohe, values, protected, predicted)
		if i == 0:
			aloss = np.array(this_aloss)*0
		aloss = aloss + np.array(this_aloss)

	floss = floss/get_avg
	aloss = aloss/get_avg
	return aloss.tolist(), aloss_labels, floss.tolist(), floss_labels 

def calculate_all(files, og_train_file, prot_var, og_test_file, val=False, all_test=False):
	global alg
	global alg_list

	samples_file_name = dir+'/samples_'+prot_var+'_'+alg+'.csv'
	if val == True:
	    samples_file_name = dir+'/samples_'+prot_var+'_'+alg+'_val.csv'
	if all_test == True:
	    samples_file_name = dir+'/samples_'+prot_var+'_'+alg+'_all_test.csv'

	#Get dataset values
	protected, _, _, predicted, _, _ = get_labels(og_file, prot_var)
	

	#Selection Parameters	
	acc_labels = ['Accuracy Gen', 'Accuracy Orig', 'Accuracy Bal Gen', 'Accuracy Bal Orig','K1 Dist']
	fair_labels = ['Disparate Impact', 'Disparate Impact Gen', 'Equal Opp. Diff', 'Statistical Par. Diff', 'Theil Index','Avg. Odds']
	theta=1
	w=1
	rewrite_file = 'False'	
	try:
		samples = pd.read_csv(samples_file_name,header='infer')
		samples = samples.set_index(['filename'],drop=True)
		file_exists = 'True'
		original_columns = samples.columns
		if (list(original_columns) != ['Algorithm']+acc_labels+fair_labels):
			file_exists = 'False'
			samples.to_csv(dir+'/old_samples_'+prot_var+'_'+alg+'.csv', index_label='filename')	
			os.remove(samples_file_name)
			
	except:
		file_exists = 'False'


	## READ ALL FILES
	if file_exists == 'True':
		files = set(files)-set(samples.index.values.tolist())
		samples.columns = list(range(0, len(original_columns)))
		samples = samples.T

	sample_num = 0
	drop_prot = ['gender','race'] 
	try:
		drop_prot.remove(prot_var)
		files = [x for x in files if drop_prot[0] not in x]
	except:
		print('no other protected')

	print(files)

	#GET TRAINING DATA FROM ORIGINAL DATASET
	og_train_data,_,ohe = get_data(og_train_file,prot_var)
	if 'Unnamed: 0' in og_train_data.columns:
		og_train_data = og_train_data.drop(columns=['Unnamed: 0'])
	#OG TEST DATA
	og_test_data,_,_ = get_data(og_test_file,prot_var)
	if 'Unnamed: 0' in og_test_data.columns:
		og_test_data = og_test_data.drop(columns=['Unnamed: 0'])
	og_test_data = fill_x(og_train_data, og_test_data).dropna().reset_index(drop=True)
	og_train_data = fill_x(og_test_data, og_train_data).dropna().reset_index(drop=True)

	if file_exists == 'False':
		#og_reg = train_on_data(og_train_data, split=False)
		og_acc, acc_labels, og_fair, fair_labels = read_files(og_train_file, og_train_file, og_test_data, og_train_data, prot_var, ohe, True)
		samples=pd.DataFrame(pd.Series([alg]+og_acc+og_fair, name='Original_0_0_0'))

	for file in files:
		print(alg)
		print(file)
		if True: #if alg in file or not any(val in file for val in alg_list):
			print(str(sample_num)+'/'+str(len(files)))
			aloss,acc_labels, floss,fair_labels = read_files(file, og_file, og_test_data, og_train_data, prot_var, ohe)
			sample = pd.Series([alg]+aloss+floss, name=file)
			samples = pd.concat([samples, sample],axis=1) 
			sample_num = sample_num+1
			to_write = samples.T
			to_write.columns = ['Algorithm']+acc_labels+fair_labels
			to_write.to_csv(samples_file_name,index_label='filename')
	samples = samples.T
	samples.columns = ['Algorithm']+acc_labels + fair_labels
	samples.to_csv(samples_file_name,index_label='filename')

if __name__ == "__main__":
	global alg
	global alg_list
	dir = sys.argv[1]
	prot_var = sys.argv[2]
	files = [f for f in os.listdir(dir) if f.startswith('sample_data_')]
	files.sort()
	f = open(dir + '//out_select_' + prot_var + '.txt','w')
	sys.stdout=f
	
	og_file, og_test, og_val, og_all_test = get_og_file(dir, test=True, valid=True, all=True)
	print('TEST FILE: ',og_test)
	print('TRAIN FILE: ',og_file)
	print('ALL TEST FILE: ',og_all_test)
	
	alg_list = ['LogReg','LinSVC','NeuralNet']#['LogReg','Percep','LinSVC','Tree','RandomForest']
	for alg in ['RandomForest']+alg_list:
	#for alg in ['LinSVC']:
	#for alg in ['RandomForest']:
		#calculate_all(files, og_file, prot_var, og_test)
		#calculate_all(files, og_file, prot_var, og_val, val=True)
		calculate_all(files, og_file, prot_var, og_all_test, all_test=True)
