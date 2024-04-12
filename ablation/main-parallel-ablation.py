import sys
import os
import warnings
import multiprocessing as mp 
import time
import math
import random
from operator import add
from sklearn.decomposition import PCA
from scipy.spatial import distance
import aif360.metrics as metrics
import aif360.datasets as datasets

# Turn on scikit-learn optimizations :
from sklearnex import patch_sklearn
patch_sklearn()

from transform_main import * 
from sklearnex.model_selection import train_test_split



def parallel_test(x):
	data = x[0]
	test_data = x[1].reset_index(drop=True)
	avg = x[2] 
	for i in range(avg):
		reg = train_on_data(data)
		pred = predict_output(test_data, reg)
		out = call_calc(test_data, pred)
		if i == 0:
			out_avg = out	
		else:
			out_avg = [ x+y for x,y in zip(out_avg, out)]
	out = [ x/avg for x in out_avg ]
	out_nopred = call_calc(data)
	return [reg, pred, out, out_nopred]
		
def get_train_test(data, split=True):
	og_x = data[data.columns[:-1]].astype(float)
	og_y = data[data.columns[-1]].astype(float)
	if not split:
		return og_x, og_y
	x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=0.33, stratify=og_y)
	return x_train, x_test, y_train, y_test

def train_on_data(data, return_test=False, return_train=False, split=False):
	#reg = RandomForestClassifier(n_estimators=500) 
	if use_gpu:
		reg = RandomForestClassifier(n_estimators=100) 
	else:
		reg = RandomForestClassifier(n_estimators=100, n_jobs=-1) 

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


	if use_gpu:
		if type(x_train) != type(cudf.from_pandas(pd.DataFrame(columns=[1]))):
			x_train_cudf = cudf.from_pandas(x_train)
			y_train_cudf = cudf.from_pandas(y_train)

	try:
		if use_gpu:
			reg.fit(x_train_cudf, y_train_cudf)#, sample_weight = weight)
		else:
			reg.fit(x_train,y_train, sample_weight = weight)
	except Exception as e:
		print("FAILED TRAINING: ",e)
		reg = -1

	if 'fnlwgt' in data.columns:
		x_train = x_train_with_weight
	if return_test:
		if use_gpu:
			test = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True).rename(og_data.columns[-1])],axis=1).reset_index(drop=True)
		else:
			test = pd.concat([x_test,y_test],axis=1).reset_index(drop=True)
		if return_train:
			if use_gpu:
				train = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(og_data.columns[-1])],axis=1).reset_index(drop=True)
			else:
				train = pd.concat([x_train,y_train],axis=1).reset_index(drop=True)
			return reg, test, train
		return reg, test 
	return reg

def predict_output(data, reg=None):
	if isinstance(data, tuple):
		reg = data[1]
		data = data[0]
	x = data[data.columns[:-1]]
	if use_gpu:
		x.reset_index(drop=True)
	else:
		x = pd.DataFrame(x).reset_index(drop=True)
	if 'fnlwgt' in x.columns:
		x_with_weight = x.copy(deep=True)
		x = x.drop(columns = ['fnlwgt'])
	if reg == -1:
		y_pred = np.zeros([len(data)])
		print('ZEROS')
	else:
		try:
			y_pred = reg.predict(x)
		except:
			y_pred = np.zeros([len(data)])
			print('ZEROS')
	if 'fnlwgt' in data.columns:
		x = x_with_weight
	return pd.concat([x,pd.Series(y_pred,name=data.columns[-1])], axis=1)


def get_start_sample(data, start_sample_size, predicted, stratified=''):
    cols = list(data.columns)
    cols.remove(predicted)
    og_x = data[cols].astype(float)
    og_y = data[predicted].astype(float)
    if stratified != '':
        x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=start_sample_size, stratify=data[stratified], shuffle=True)
    else:
        x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=start_sample_size, shuffle=True)
    return pd.concat([x_train,y_train], axis=1), pd.concat([x_test, y_test], axis=1)

def fill_x(og_data, data):
    diff = list(set(og_data.columns)-set(data.columns))
    add_cols = pd.DataFrame(0, index=np.arange(len(data)), columns=diff)
    data = pd.concat([data,add_cols],axis=1)
    return data[og_data.columns]

def call_calc(new,predicted_data=''):
    fno = False
    if type(predicted_data) == type(''):
        predicted_data = new
        fno = True

    priv_groups = [{protected: 1}]
    unpriv_groups = [{protected: 0}]

    ## ORGANIZE ORIGINAL DATA
    og_dataset= datasets.BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=new, label_names=[predicted], protected_attribute_names=[protected])
    ###ORGANIZE DATA TRAINED ON GENERATED AND TESTED ON ORIGINAL TEST DATA
    og_predicted_dataset = datasets.BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=predicted_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = None)
    og_classMetrics = metrics.ClassificationMetric(og_dataset, og_predicted_dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)# Predicted Original data on generated trained classifer


    f = min(og_classMetrics.disparate_impact(), 1/(og_classMetrics.disparate_impact()))
    if fno:
        return f

    a = 0.5*(og_classMetrics.true_positive_rate()+og_classMetrics.true_negative_rate())
    EqOp = 1-abs(og_classMetrics.equal_opportunity_difference())

    return [f, a, EqOp]

    
def train(directory, protected, label, alpha, to_select, beta, set_perc, hyp_disp, min_acc_bound, avail_num, ablation_var ):
    ## GET ORIGINAL VARIABLES
    train_start = time.time()
    #files = [f for f in os.listdir(directory) if f.endswith('.csv')]# if f.startswith('sample_data_')]
    files = [f for f in os.listdir(directory) if f.startswith('sample_data_')]
    print(files)
    file_num = len(files)
    priv = 1
    unpriv = 0
    pref = 1
    unpref = 0 
    
    ######## START WITH RANDOM DATA #######
    i = 0
    for file in files:
        data,_,_ = get_data(directory+"/"+file, og_file, protected, privileged, predicted, preferred)
        data = fill_x(og_data, data) 
        
        ## Combine all data into one dataset
        if i == 0:
            all_data = data
        else:
            all_data = pd.concat([all_data, data], axis=0, ignore_index=True) 
        i = i + 1
    all_data = all_data.astype(float).dropna(axis='index').reset_index(drop=True)
    N = int(len(all_data.index)/len(files)) 
    print(all_data.value_counts([protected,predicted]))
    val_counts = all_data.value_counts([protected,predicted])
    print("MIN COUNTS: ",min(val_counts))
    

    if avail_num != "all":
        print("AVAIL_NUM: ",avail_num)
        print("N: ",N)
        min_count = (N*avail_num) 
        min_count_mult = min_count/min(val_counts)
        print("DESIRED MIN COUNT: ", min_count)
        print("MIN MULT: ",min_count_mult)
        print("MIN MULT * N: ",min_count_mult*N)
        all_data_avail = pd.DataFrame(columns=all_data.columns)
        for idx in [(0,0),(0,1),(1,0),(1,1)]:
            print(idx)
            a,b = idx
            this_vc = val_counts.loc[idx]
            print("THIS VC: ",this_vc)
            print("MIN MULT * THIS COUNT: ",min_count_mult*this_vc)
            all_data_avail = all_data_avail.append(all_data[(all_data[protected]==a) & (all_data[predicted]==b)].sample(int(min_count_mult*this_vc)))
        all_data = all_data_avail
        print("LEN(INDX): ",len(all_data.index))
        if len(all_data.index) < N:
            print("NOT ENOUGH DATA AFTER AVAIL")
            exit()

    to_select = int(N*to_select) #Transform percent of select to number
    set_num = int(N*set_perc)
    if set_num > to_select:
        set_num = to_select
    print("FINAL DATSET SIZE: ",N)
    print("NUMBER OF POINTS TO CHOOSE: ",to_select)
    print("SET SIZE: ",set_num)
    if N < 10000:
        avg=3
    else:
        avg=1


    start_sample_size = max(N - to_select, 12)# set_num)
    if 12 > N-to_select:
        to_select = to_select-(12-(N-to_select))
    print("START SAMPLE SIZE: ", start_sample_size)
    print("NUMBER OF POINTS TO CHOOSE: ",to_select)
    all_data, new_dataset = get_start_sample(all_data, start_sample_size, predicted, [protected,predicted]) 
    all_data = all_data.reset_index(drop=True)
    # TEST DATA IS FROM SYNTHETIC DATASET
    '''
    all_data, test_data = get_start_sample(all_data, max(N,1000), predicted, [protected,predicted])
    test_data = test_data.reset_index(drop=True)
    all_data = all_data.reset_index(drop=True)
    all_data = all_data.sample(frac=1)
    '''
    # TEST DATA IS FROM REAL DATASET
    #_, test_data = get_start_sample(og_data, int(len(og_data.index)/3), predicted, [protected,predicted]) 
    test_data = og_data
    all_data = all_data.sample(frac=1).reset_index(drop=True)
    ######## END START WITH RANDOM SAMPLE ########
   

    #### INITIALIZE VARIABLES ####
    i = 0
    if to_select != 0:
        to_select = math.ceil(to_select/set_num)
    i_round = 0
        
    #### GET VALIDATION SET ACCURACY AND FAIRNESS ####
    start_acc = 0
    start_disp = 0
    start_avg = 5
    for i in range(start_avg):
        ## IF TEST DATA IS FROM REAL DATASET
        train_i_data, test_i_data = get_start_sample(og_data, int(len(og_data.index)/3), predicted, [protected,predicted]) 
        train_i_data = train_i_data.reset_index(drop=True)
        test_i_data = test_i_data.reset_index(drop=True)
        reg = train_on_data(train_i_data, split=True)
        pred = predict_output(test_i_data, reg)
        start_vals = call_calc(test_i_data, pred)
        # END # 
        
        ''' 
        ## IF TEST DATA IS FROM SYNTHETIC DATASET
        reg = train_on_data(all_data, split=True)
        pred = predict_output(test_data, reg) 
        start_vals = call_calc(test_data, pred) 
        # END # 
        '''

        start_acc = start_acc + start_vals[1] #Original accuracy
        start_disp = start_disp + start_vals[0]
    start_acc = start_acc/start_avg
    start_disp = start_disp/start_avg
    fno = call_calc(all_data)
    print('Start Acc: ', start_acc)
    print('Start Disp: ', 1-start_disp)
    print('Start Ground Disp: ', 1-fno)


    #### GET STARTING SAMPLE ACCURACY AND FAIRNESS ####
    a_test = start_acc 
    try_num = 15 
    progress = pd.DataFrame(columns=[ 'Select Fair', 'Select G Fair', 'Select Acc' ,'Hyp', 'Best Res', 'time'])
    progress.loc[0] = [1-start_disp, 1-fno, start_acc, 0, 0, 0]
    cats = [(0,0),(0,1),(1,0),(1,1)]
    final_directory = 'ablation_study_'+ablation_var+'/'+data_name+'_selected_samples'
    if not os.path.isdir(final_directory):
        os.makedirs(final_directory) 
    
    ## SELECT
    while i_round < to_select: # or i_round < 30:
       set_size = set_num
       print("NEW DATSET SIZE: ", len(new_dataset.index))
       if i_round + 1 == to_select: # LAST ROUND
           last_set_num = N-len(new_dataset.index)
           set_size = last_set_num	
           print("LAST SET SIZE: ",set_size)
       #_, test_data = get_start_sample(og_data, int(len(og_data.index)/3), predicted, [protected,predicted]) 
       start_time = time.time()
       a_curr = a_test
       try_data_list = []
       for j in range(try_num):
           try_data = pd.DataFrame(columns=all_data.columns)
           nums = [0]*len(cats)
           for n in range(len(cats)-1): 
               nums[n] = random.randint(0, max(set_size-sum(nums), 0))
           nums[-1] = set_size-sum(nums)
           random.shuffle(nums)
           if sum(nums) != set_size:
               print("NUMS != SET_SIZE: ",sum(nums))
               exit()
           i = 0
           for a,b in cats:
              try:
                  try_data = try_data.append(all_data[(all_data[protected]==a) & (all_data[predicted]==b)].sample(nums[i]))
              except:
                  if len(all_data[(all_data[protected]==a) & (all_data[predicted]==b)].index) < 1:
                      cats.remove((a,b))
                      print("EXHAUSTED: Prot="+str(a)+", Pred="+str(b))
                  else:
                      try_data = try_data.append(all_data[(all_data[protected]==a) & (all_data[predicted]==b)])
              i += 1
           if len(try_data.index) < set_size:
               try_data = try_data.append(all_data.sample(set_size - len(try_data.index)))
           try_data_list = try_data_list + [try_data] 
       ## TEST NEIGHBORS
       if not use_gpu:
           with mp.Pool() as pool:
               pool_res = pool.map(parallel_test, [(new_dataset.append(x, ignore_index=True), test_data, avg) for x in try_data_list])
           pool_reg, pool_predicted, pool_out, pool_out_nopred = map(list, zip(*pool_res))
           floss, aloss, EqOp = map(list, zip(*pool_out))
       else:
           i = 0
           for try_data in try_data_list:
               pool_res = parallel_test((new_dataset.append(try_data, ignore_index=True), test_data, avg))
               if i == 0:
                   pool_reg = [pool_res[0]]
                   pool_predicted = [pool_res[1]]
                   pool_out = [pool_res[2]]
                   pool_out_nopred = [pool_res[3]]
               else:
                   pool_reg.append(pool_res[0])
                   pool_predicted.append(pool_res[1])
                   pool_out.append(pool_res[2])
                   pool_out_nopred.append(pool_res[3])
               i = i + 1

       floss_nopred = pool_out_nopred

       # Mask nan values
       floss_nan = np.isnan(floss)
       aloss_nan = np.isnan(aloss)
       mask = np.logical_or(floss_nan, aloss_nan)
       disp= np.ma.masked_array(floss, mask=mask)
       disp_nopred = np.ma.masked_array(floss_nopred, mask=mask) 
       bal = np.ma.masked_array(aloss, mask=mask)  
       EqOp = np.ma.masked_array(EqOp, mask=mask)
       EqOp_disp = np.ma.masked_array( [ 1/2*(EqOp[i] + disp[i]) for i in range(len(disp)) ], mask=mask )
       
       ## CHOOSE BEST NEIGHBOR
       #min_acc = max(start_acc-min_acc_bound, .60)
       min_acc = start_acc-min_acc_bound
       exp = min_acc - a_curr
       #hyp = beta**( exp )
       hyp = 10**(beta*exp )

       if hyp_disp == "disp":
           disp_try = disp
       elif hyp_disp == "disp_nopred":
           disp_try = disp_nopred
       elif hyp_disp == "both":
           disp_try = np.divide( np.add( disp, disp_nopred ), 2 )
       elif hyp_disp == "None":
           disp_try = np.array([ 0 for x in disp ])
       elif hyp_disp == "EqOp":
           disp_try = EqOp
       elif hyp_disp == "EqOp_and_Disp":
           disp_try = EqOp_disp

       res = 1/hyp*( .5*disp_try ) + hyp*bal
       best_ind = np.argmax( res ) 
       best_res = res[best_ind]
       choice = str(hyp)
         
       # UPDATE DATASET
       try_data = try_data_list[best_ind]	
       new_dataset = new_dataset.append(try_data, ignore_index=True)

       ind_to_drop = try_data.index.tolist()
       #print('index: ',ind_to_drop)
       all_data.drop(index=ind_to_drop, inplace=True)
       all_data = all_data.reset_index(drop=True)

       fno_test = 1-floss_nopred[best_ind]
       f_test = 1-floss[best_ind]
       a_test = aloss[best_ind]

       # SAVE PROGRESS AND DATASET
       i_round = i_round + 1
       t = time.time() - start_time 
       
       save_progress_time = time.time()
       progress.loc[len(progress.index)] = [f_test, fno_test, a_test, hyp, best_res, t] 

       progress.to_csv(final_directory+'/'+data_name+'_'+protected+'_'+str(label)+'_progress.csv')
       print(pd.DataFrame(progress.iloc[len(progress.index)-1]).T)
       

    file_name = data_name+'_'+protected+'_'+str(label)
    final_file_name = final_directory+'/sample_data_parallel_'+file_name 
    save_dataset(new_dataset.copy(), final_file_name, protected, ohe, og_file, i_round)

def save_dataset(new_dataset_save, file_name, protected, ohe, og_file, i_round):
    ## Get Final Dataset
    new_dataset_save = new_dataset_save.reset_index(drop=True)
    new_dataset_save = return_data(new_dataset_save, ohe, og_file, protected, predicted, privileged, preferred)
    #new_dataset_save = new_dataset_save.rename(columns={"protected":protected,"predicted":predicted})
    #new_dataset_save[protected]=np.where(new_dataset_save[protected]==1, 'privileged', 'unprivileged')
    #new_dataset_save[predicted]=np.where(new_dataset_save[predicted]==1, 'preferred', 'unpreferred') 
    new_dataset_save.to_csv(file_name+'.csv', index=False)	
    return 

#################################################### MAIN ######################################################
if __name__ == "__main__":
    ############################
    # Get Data + One Hot Encode
    ############################
    global gen_reg
    use_gpu = False
    #use_gpu = True
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_columns',None)
    try:
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


        # Get all Files
        og_file = training_file
        og_data,_,ohe = get_data(og_file, og_file, protected, privileged, predicted, preferred)


        select_percent = float(1)
        beta = 10
        set_perc = float(5)/100  #(or 10?)
        hyp_disp = "disp"
        min_acc_bound = 0 
        avail_num = "all"
        
        if len(sys.argv) > 9:
            if sys.argv[8] == "-b":
                beta = float(eval(sys.argv[9]))
                ablation_var = "beta"
            elif sys.argv[8] == "-sp":
                set_perc = float(sys.argv[9])/100
                ablation_var = "set_percent_start_acc"
            elif sys.argv[8] == "-p":
                select_percent = float(sys.argv[9])
                ablation_var = "select_percent"
            elif sys.argv[8] == "-d":
                hyp_disp = sys.argv[9]
                ablation_var = "hyp_disp"
            elif sys.argv[8] == "-a":
                min_acc_bound = float(sys.argv[9])
                ablation_var = "min_acc_bound"
            elif sys.argv[8] == "-n":
                avail_num = float(sys.argv[9])
                ablation_var = "avail_num"
        else:
            ablation_var = "GAN_type"

        print('Ablation: ',ablation_var)
        print('Beta: ',beta)
        print('Set Percent: ',set_perc)
        print('Select_percent: ',select_percent)
        print('Hyp Disp: ',hyp_disp)
        print('Min Acc Bound: ',min_acc_bound)
        print('Available Number: ',avail_num)


        if use_gpu:
            from cuml.ensemble import RandomForestClassifier
            import cudf
            os.environ['CUDA_VIDIBLE_DEVISES']="0,1"
        else:
            from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        print(e)
        print('Please supply the following arguments: [directory] [dataset_name] [training_file] [protected feature] [privileged value] [predicted feature] [preferred value] [flag] [beta/set_perc/selected_data_percentage (ex. .5)] ]')  
        exit()
    
    alpha = 1

    for label in ['1','2','3']:
        train(directory, protected, label, alpha, select_percent, beta, set_perc, hyp_disp, min_acc_bound, avail_num, ablation_var)
