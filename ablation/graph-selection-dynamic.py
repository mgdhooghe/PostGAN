import math
import scipy.spatial.distance as distance
import sys
import os
import pandas as pd
import random
import operator
from transform_main import *
import warnings
import time

def fill_x(og_data, data):
    diff = list(set(og_data.columns)-set(data.columns))
    add_cols = pd.DataFrame(0, index=np.arange(len(data)), columns=diff)
    data = pd.concat([data,add_cols],axis=1)
    return data[og_data.columns]

def d(u,s):
	#return distance.cdist([u],[s])
	return distance.cdist(u,s)

def update_min_dict(U,X):
	#### FOR EACH POINT IN U GET THE DISTANCE TO THE CLOSEST POINT IN S
	## GET DISTANCE TO POINTS IN U FROM ALL POINTS IN SELECTED SET X
	temp = d(U,X)
	#### ADD THE VALUE FROM U THAT HAS THE MAXIMUM MINIMUM DISTANCE TO A POINT IN S
	## GET MINIMUM DISTANCE TO A POINT IN X FOR EACH POINT IN U 
	min_indx = X.index.values[np.argmin(temp, axis=1)] 
	## GET DISTANCE OF CLOSEST X POINT FOR EACH POINT IN U 
	min_d = np.min(temp, axis=1) 
	## ZIP X INDEX AND ITS DISTANCE FOR EACH POINT IN U 
	min_d = tuple(zip(min_indx,min_d))
	## GET INDEX OF EACH POINT IN X
	#min_s = U.index.values 
	## KEY IS INDEX OF U, VALUE IS (X INDEX, DISTANCE)
	new_min_ds = dict( map(lambda i,j: (i,j), U.index.values, min_d) )
	return new_min_ds

def GMM(U,I,k):
	global data_name
	global training_file
	global ohe
	global protected
	global predicted
	global privileged
	global preferred
	try:
		print('U Size: ',U.shape)
	except:
		print('U is None')
	try:
		print('I Size: ',I.shape)
	except:
		print('I is None')
	S = None
	if not isinstance(I,pd.DataFrame):
		t = random.randint(0,len(U.index)-1)
		S = pd.DataFrame(U.loc[t]).T#.reset_index(drop=True).T
		S.columns=U.columns
		U = U.drop([t])#.reset_index(drop=True)
	if isinstance(I,pd.DataFrame):
		if len(I.index) < 1:
			t = random.randint(0,len(U.index)-1)
			S = pd.DataFrame(U.loc[t]).T#.reset_index(drop=True).T
			S.columns=U.columns
			U = U.drop(index=[t])#.reset_index(drop=True)
	try:
		print('S Size: ',S.shape)
	except:
		print('S is None')

	## INITIALIZE DICTIONARY
	min_ds = {}
	if not isinstance(I,pd.DataFrame):
		S_I = S
	elif not isinstance(S,pd.DataFrame):
		S_I = I
		S = pd.DataFrame(columns=U.columns)
	else:
		S_I = pd.concat([S,I],axis=0,ignore_index=True).reset_index(drop=True)


	start = True
	start_time = time.time()
	while len(S.index) < k and len(U.index) > 0:
		round_start_time = time.time()
		## CALCULATE MINIMUM DISTANCE
		if start:
			print("START")
			min_ds = update_min_dict(U,S_I)
			start = False
		else:
			new_min_ds = update_min_dict(U,X)
			## KEEP MIN FOR OVERLAP 
			min_ds = { k : (min_ds.get(k) if min_ds.get(k)[1] < new_min_ds.get(k)[1] else new_min_ds.get(k)) for k in new_min_ds.keys() }
		## GET U POINT WITH MAXIMUM MINIMUM DISTANCE TO ALL POINTS IN X
		max_U = max(min_ds, key=lambda k: min_ds[k][1])
		X = pd.DataFrame(U.loc[max_U]).T
		## ADD POINT FROM U TO S
		S = pd.concat([S,X],axis=0)
		## DROP ADDED POINT FROM U
		U = U.drop(index=[max_U])
		## DROP ADDED POINT FROM Min_ds
		del min_ds[max_U]
		if len(S.index) % 50 == 0:
			print('S: ',S)
			print('Round Time: ',time.time() - round_start_time)
			print('Total Time: ',time.time() - start_time)


	return S

def FAIR_SWAP(U1,U2,k1,k2,protected):
	U = pd.concat([U1,U2])
	k = k1+k2
	if len(U2.index) + len(U2.index) < k:
		print('NOT ENOUGH DATA')
		exit()
	if len(U1.index) < k1:
		k1 = len(U1.index)
		k2 = k - k1
		print('UPDATED k1: ',k1)
	if len(U2.index) < k2:
		k2 = len(U2.index)
		k1 = k - k2
		print('UPDATED k2: ',k2)
	#Color-Blind Phase
	S = GMM(U, None, k)
	print('First S: ',S)
	# SAVE BEFORE FAIR
	file_name = 'FAIR_SWAP/'+data_name+'-nofair-ohe-'+str(label)+'.csv'
	S.to_csv(file_name, index=False)
	S_ret = return_data(S.copy(deep=True).reset_index(drop=True), ohe, training_file, protected, predicted, privileged, preferred)
	file_name = 'FAIR_SWAP/sample_data_'+data_name+'-nofair-FAIR-SWAP-'+str(label)+'.csv'
	S_ret.to_csv(file_name, index=False)
	# SPLIT S INTO PROTECTED GROUPS
	#S1 = pd.merge(S,U1, how='inner')
	S1 = S[S[protected] == 0]
	U1 = pd.merge(U1,S1, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
	#S2 = pd.merge(S,U2, how='inner')
	S2 = S[S[protected] == 1]
	U2 = pd.merge(U2,S2, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
	#Balancing Phase
	if k1-len(S1.index) > k2-len(S2.index):
		Uu = U1
		Su = S1
		ku = k1 
		ko = k2 
		Uo = U2
		So = S2
	else:
		Uu = U2	
		Su = S2
		ku = k2 
		ko = k1 
		Uo = U1
		So = S1

	
	if ku-len(Su.index) > 0:
		E = GMM(Uu, Su, ku-len(Su.index)).reset_index(drop=True)
		E_compare = E.copy(deep=True)
		print('E size: ',len(E_compare.index))
		print('So size: ',len(So.index))

		while len(So.index) > ko:
			## FOR EACH DATAPOINT IN E FIND THE CLOSEST DATAPOINT IN SO
			min_ds = update_min_dict(So,E_compare)
			## GET UNIQUE CLOSEST VALUES
			closest_points = [ value[0] for value in min_ds.values() ]
			unique_keys = [key for key, value in min_ds.items() if closest_points.count(value[0]) == 1 ]
			unique_points = unique_keys #[value[0] for value in [min_ds[x] for x in unique_keys]]
			## REMOVE UNIQUE CLOSEST VALUES
			So = So.drop(index=unique_points)
			for x in unique_keys:
				E_compare = E_compare.drop(index=[min_ds[x][0]])
				del min_ds[x]
			## FOR EACH OVERLAPPING DATAPOINT IN E FIND THE MIN CLOSEST DATAPOINT IN SO
			closest_X_points = set([ value[0] for value in min_ds.values() ]) 
			for closest_point in closest_X_points:
				min_dist = { k : v for k,v in min_ds.items() if v[0] == closest_point } #Get all U with distances closest to this X
				min_key = min(min_dist, key=lambda k: min_ds[k][1]) # Get U value closest to this X
				So = So.drop(index=[min_key]) # Remove closest U value
				del min_ds[min_key]
				## REPEAT UNTIL NO OVERLAPPING DATAPOINTS
				E_compare = E_compare.drop(index=[closest_point])
			print('E size: ',len(E_compare.index))
			print('So size: ',len(So.index))
		
		So = So.reset_index(drop=True)
		Su = pd.concat([Su,E]).reset_index(drop=True)
		print('Su: ',Su)
		print('So: ',So)
	final = pd.concat([Su,So]).reset_index(drop=True)
	print('Final: ',final)
	return final 

if __name__ == "__main__":
	global data_name
	global training_file
	global ohe
	global protected
	global predicted
	global privileged
	global preferred
	global label

	directory = sys.argv[1]
	print('Synthetic Data Directory: ',directory)
	data_name = sys.argv[2]
	print('Dataset Name: ',data_name)
	training_file = sys.argv[3]
	print('Training Dataset File: ',training_file)
	protected = sys.argv[4]
	print('Protected Feature: ',protected)
	privileged = sys.argv[5]
	print('Privileged Value: ',privileged)
	predicted = sys.argv[6]
	print('Predicted Feature: ',predicted)
	preferred = sys.argv[7]
	print('Preferred Value: ',preferred)

	files = files = [f for f in os.listdir(directory) if f.startswith('sample_data_')]
	file_num = len(files)

	######## START WITH RANDOM DATA #######
	og_data,_,ohe = get_data(training_file, training_file, protected, privileged, predicted, preferred)
	i = 0
	print(files)
	for file in files:
		data,_,_ = get_data(directory+"/"+file, training_file, protected, privileged, predicted, preferred)
		data = fill_x(og_data, data) 
		
		## Combine all data into one dataset
		if i == 0:
		    all_data = data
		else:
		    all_data = pd.concat([all_data, data], axis=0, ignore_index=True) 
		i = i + 1
	all_data = all_data.astype(float).dropna(axis='index').reset_index(drop=True)
	all_data_unpriv = all_data[all_data[protected] == 0]
	all_data_priv = all_data[all_data[protected] == 1]
	k_2 = len(og_data.index)/2
	for label in [0,1,2]:
		if len(all_data_unpriv.index) < k_2 or len(all_data_priv.index) < k_2:
			print("NOT ENOUGH DATA")
			exit()
		S = FAIR_SWAP(all_data_unpriv, all_data_priv, k_2, k_2, protected)
		file_name = 'FAIR_SWAP/'+data_name+'-ohe-'+str(label)+'.csv'
		S.to_csv(file_name, index=False)
		S = return_data(S, ohe, training_file, protected, predicted, privileged, preferred)
		file_name = 'FAIR_SWAP/sample_data_'+data_name+'-FAIR-SWAP-'+str(label)+'.csv'
		S.to_csv(file_name, index=False)
