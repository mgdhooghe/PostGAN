import random
import math
import sys
import os
import warnings
import multiprocessing as mp 
import time
from operator import add
from sklearn.ensemble import RandomForestClassifier

from transform_main import * 
from sklearnex.model_selection import train_test_split


def get_train_test(data, split=True):
	og_x = data[data.columns[:-1]].astype(float)
	og_y = data[data.columns[-1]].astype(float)
	if not split:
		return og_x, og_y
	x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=0.33, stratify=og_y)
	return x_train, x_test, y_train, y_test

def train_on_data(data, return_test=False, return_train=False, split=True):
	reg = RandomForestClassifier(n_estimators=500) 

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
		reg.fit(x_train,y_train, sample_weight = weight)
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
    def get_acc(predicted_data, new):
        # Calculate Balanced Accuracy
        ## All y
        new_y = new[predicted]
        pred_y = predicted_data[predicted]
        
        ## Y count
        pos = sum(new_y)
        tru_p = sum(pred_y[new_y==1])
        if pos != 0:
            true_pos = tru_p/pos 
        else:
            true_pos = 0
   
        ## Y Neg count
        neg = len(new_y)-pos
        tru_n = len(pred_y[new_y==0])-sum(pred_y[new_y==0])  
        if neg != 0:
            true_neg = tru_n/neg 
        else:
            true_neg = 0
        a = 0.5*(true_pos+true_neg)
        a = a
        if pred_y[0] == -1 or sum(pred_y) == 0:
            a = 0
        return a
    
    def get_disp(predicted_data):
        # Calculate Disparate Impact
        priv = 1
        unpriv = 0
        pref = 1
        unpref = 0 
        
        val_count = predicted_data[[protected,predicted]].value_counts()
        indices = ((unpriv, pref), (priv, pref), (unpriv, unpref), (priv, unpref))
        val_count = val_count.reindex(indices, fill_value=0)
        un_pref = val_count[(unpriv, pref)]
        priv_pref = val_count[(priv, pref)]
        un_unpref = val_count[(unpriv, unpref)]
        priv_unpref = val_count[(priv, unpref)]
        if (priv_pref + priv_unpref) == 0:
            f = 0 #math.inf 
        elif (un_pref + un_unpref) == 0:
            f = 0 #math.inf 
        else:
            fa = priv_pref/(priv_pref+priv_unpref)
            fb = un_pref/(un_pref+un_unpref)
            f = min(fa/fb,fb/fa)
        
        if math.isnan(f):
            f = 0
        return f

    if type(predicted_data) == type(''):
        f = get_disp(new)
        return f

    a = get_acc(predicted_data,new)
    f = get_disp(predicted_data)

    return [f, a]

    
def train(directory, protected, label, alpha, to_select ):
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
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        data = fill_x(og_data, data) 
        
        ## Combine all data into one dataset
        if i == 0:
            all_data = data
        else:
            all_data = pd.concat([all_data, data], axis=0, ignore_index=True) 
        i = i + 1
    all_data = all_data.astype(float).dropna(axis='index')

    N = len(og_data.index)
    if N < 10000:
        set_num = 25 
        avg = 3
    else:
        set_num = 50
        avg = 1

    start_sample_size = max(N - to_select, set_num)
    all_data, new_dataset = get_start_sample(all_data, start_sample_size, predicted, [protected,predicted]) 
    all_data = all_data.reset_index(drop=True)
    all_data, test_data = get_start_sample(all_data, max(N,1000), predicted, [protected,predicted])
    test_data = test_data.reset_index(drop=True)
    all_data = all_data.reset_index(drop=True)
    all_data = all_data.sample(frac=1)
    ######## END START WITH RANDOM SAMPLE ########
   

    #### INITIALIZE VARIABLES ####
    beta = 10
    i = 0
    to_select = int(to_select/set_num)+1
    selected_count = to_select
    round = 0
        
    #### GET VALIDATION SET ACCURACY AND FAIRNESS ####
    start_acc = 0
    start_disp = 0
    start_avg = 5
    for i in range(start_avg):
        reg = train_on_data(all_data)
        pred = predict_output(test_data, reg) 
        start_vals = call_calc(test_data, pred) 
        start_acc = start_acc + start_vals[1] #Original accuracy
        start_disp = start_disp + start_vals[0]
    start_acc = start_acc/start_avg
    start_disp = start_disp/start_avg
    fno = call_calc(all_data)


    #### GET STARTING SAMPLE ACCURACY AND FAIRNESS ####
    f_test = 1
    a_test = 1 
    try_num = 10 
    noMore = 0
    
    ## SELECT
    while round < to_select or round < 30: #len(new_dataset.index) < len(og_data.index):
       start_time = time.time()
       out_nopred = []
       out = []
       a_curr = a_test
       indicies = []
       for j in range(try_num):
           try_data = pd.DataFrame(columns=all_data.columns)
           nums = np.random.random_sample((4,))
           nums = (nums/nums.sum()*set_num).astype(int)
           cats = [(0,0),(0,1),(1,0),(1,1)]
           i = 0
           for a,b in cats:
              try:
                  try_data = try_data.append(all_data[(all_data[protected]==a) & (all_data[predicted]==b)].sample(nums[i]))
              except:
                  try: 
                      print("NOT ENOUGH: Prot="+str(a)+", Pred="+str(b))
                      try_data = try_data.append(all_data[(all_data[protected]==a) & (all_data[predicted]==b)])
                      noMore = 'Prot: ' + str(a) + ',' + 'Pred: ' + str(b) 
                  except:
                      print("NO MORE: Prot="+str(a)+", Pred="+str(b))
              i += 1
           #_, try_data =  get_start_sample(all_data, set_num, predicted)
           indicies.append(np.array(try_data.index))
           ## TEST NEIGHBORS
           # TEST
           out_avg = [0,0]
           for x in range(avg):
               try_gen_reg_all = train_on_data(new_dataset.append(try_data,ignore_index=True), split=False)
               try_predicted = predict_output(test_data, try_gen_reg_all) 
               out_x = call_calc(test_data, try_predicted)
               out_avg = list( map(add, out_avg, out_x) )
           out.append(np.divide(out_avg,avg)) 
           # NOPRED TEST
           out_nopred.append(call_calc(new_dataset.append(try_data,ignore_index=True)))
       floss_nopred = out_nopred
       floss, aloss = map(list, zip(*out)) 

       # Mask nan values
       floss_nan = np.isnan(floss)
       aloss_nan = np.isnan(aloss)
       mask = np.logical_or(floss_nan, aloss_nan)
       disp= np.ma.masked_array(floss, mask=mask) 
       disp_nopred = np.ma.masked_array(floss_nopred, mask=mask) 
       bal = np.ma.masked_array(aloss, mask=mask)  
       
       ## CHOOSE BEST NEIGHBOR
       min_acc = max(start_acc-.10, .60)
       exp = -(a_curr - min_acc)
       hyp =  beta**( 5*exp )
       res = 1/hyp*np.add( disp,disp_nopred ) + hyp*bal
       best_ind = np.argmax( res ) 
       choice = str(hyp)
         
       # UPDATE DATASET
       ind = i + best_ind*set_num 
       i = ind + 1
       try_data = all_data.loc[indicies[best_ind]]
       new_dataset = new_dataset.append(try_data, ignore_index=True)

       all_data.drop(index=try_data.index.tolist(), inplace=True)
       all_data = all_data.reset_index(drop=True)

       fno = 1-floss_nopred[best_ind]
       f_test = 1-floss[best_ind]
       a_test = aloss[best_ind]
       print('Fairness: ',f_test)
       print('Accuracy: ',a_test)
       # SAVE PROGRESS AND DATASET
       round = round + 1
       t = time.time() - start_time 
       
       print("Round Time: ", time.time()-start_time)
       print("Total Time: ", time.time()-train_start)


    final_directory = directory+'/'+data_name+'_selected_samples'

    if not os.path.isdir(final_directory):
        os.makedirs(final_directory) 

    file_name = protected+'_'+str(label)
    final_file_name = final_directory+'/sample_data_'+file_name 
    save_dataset(new_dataset.copy(), final_file_name, protected, ohe, og_file, round)

def save_dataset(new_dataset_save, file_name, protected, ohe, og_file, round):
    ## Get Final Dataset
    new_dataset_save = return_data(new_dataset_save, ohe)
    new_dataset_save = new_dataset_save.rename(columns={"protected":protected,"predicted":predicted})
    new_dataset_save[protected]=np.where(new_dataset_save[protected]==1, 'privileged', 'unprivileged')
    new_dataset_save[predicted]=np.where(new_dataset_save[predicted]==1, 'preferred', 'unpreferred') 
    new_dataset_save.to_csv(file_name+'.csv')	
    return 

#################################################### MAIN ######################################################
if __name__ == "__main__":
    ############################
    # Get Data + One Hot Encode
    ############################
    global gen_reg
    warnings.filterwarnings("ignore")
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
        select_percent = sys.argv[8]
        print('Percent of Data Selected: ',select_percent)
        select_percent = float(select_percent)
    except:
        print('Please supply the following arguments: [directory] [dataset_name] [training_file] [protected feature] [privileged value] [predicted feature] [preferred value] [selected_data_percentage (ex. .5)]')  
        exit()
    
    alpha = 1

    # Get all Files
    og_file = training_file
    og_data,_,ohe = get_data(og_file, og_file, protected, privileged, predicted, preferred)

    total_selected = len(og_data.index)

    for label in ['0','1','2','3','4','5','6']:
        for to_select in [total_selected*select_percent]:
            train(directory, protected, label, alpha, int(to_select))
