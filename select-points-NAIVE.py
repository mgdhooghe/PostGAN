import random
import math
import sys
import os
import warnings
import multiprocessing as mp 
import time
from operator import add

from transform_data_dataset import *
from sklearnex.model_selection import train_test_split#, StratifiedShuffleSplit


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
            print("ALL 0")
   
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

    
def train(directory, prot_var, retrain, label, alpha, to_select, alg):
    ## GET ORIGINAL VARIABLES
    train_start = time.time()
    files = [f for f in os.listdir(directory) if f.startswith('sample_data_')]
    file_num = len(files)
    protected, _, _, predicted, _, _ = get_labels(og_file, prot_var)
    priv = 1
    unpriv = 0
    pref = 1
    unpref = 0 


    if 'adult' in directory:
        data_name = 'adult'
    elif 'german' in directory:
        data_name = 'german'
    elif 'compas' in directory:
        data_name = 'propublica_compas'
    elif 'census' in directory:
        data_name = 'census'
    elif 'bank' in directory:
        data_name = 'bank'
    elif 'medical' in directory:
        data_name = 'medical'
    else:
        data_name = ''

    final_directory = directory+'/'+data_name+'_selected_samples'

    if not os.path.isdir(final_directory):
        os.makedirs(final_directory) 

    
    ######## START WITH RANDOM DATA #######
    i = 0
    N = len(og_data.index)
    for file in files:
        print(file)
        data,_,_ = get_data(directory+"/"+file, prot_var)
        print(data)
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
    print('ALL: ',all_data[[protected,predicted]].value_counts())
    
    #### SAVE ALL GAN DATA ####
    file_name = 'ALL_GAN_DATA'
    final_file_name = final_directory+'/sample_data_'+file_name 
    p = mp.Process(target=save_dataset, args=(all_data.copy(), final_file_name, prot_var, ohe, og_file, og_data_og, round, 'sample_data'))
    p.start()

    #### GET TEST DATA ####
    all_data, test_data = get_start_sample(all_data, max(N,1000), predicted, [protected,predicted])
    test_data = test_data.reset_index(drop=True)
    all_data = all_data.reset_index(drop=True)
    all_data = all_data.sample(frac=1)

    #### GET TRY ACCURACY AND FAIRNESS ####
    reg = train_on_data(all_data, new_alg=alg)
    pred = predict_output(test_data, reg) 
    all_disp, all_acc = call_calc(test_data, pred) 
    print('All disp: ',all_disp)

    best = 0
    for i in range(500):
        if i%10 == 0:
            print('ROUND: ',i)
        _, try_dataset = get_start_sample(all_data, N, predicted) 
 
        #### GET TRY ACCURACY AND FAIRNESS ####
        reg = train_on_data(try_dataset, new_alg=alg)
        pred = predict_output(test_data, reg) 
        try_disp, try_acc = call_calc(test_data, pred) 
        try_fno = call_calc(all_data)
        select = try_acc + try_disp
        if best < select:
            best = select 
            best_dataset = try_dataset
            print('All Disp: ',all_disp)
            print('All Acc: ',all_acc)
            print('Try Disp: ',try_disp)
            print('Try Acc: ',try_acc)


    best_dataset = best_dataset.reset_index(drop=True)
    print('BEST DATASET: ',best_dataset) 
    file_name = prot_var+'_NAIVE__'+alg+'_'+str(label)
    final_file_name = final_directory+'/sample_data_'+file_name 
    p = mp.Process(target=save_dataset, args=(best_dataset.copy(), final_file_name, prot_var, ohe, og_file, og_data_og, round, 'sample_data'))
    p.start()

def save_dataset(new_dataset_save, file_name, prot_var, ohe, og_file, og_data_og, round, start=''):
    global alg
    ## Get Final Dataset
    new_dataset_save = return_data(new_dataset_save, ohe, og_data_og)
    print('Before priv/prot: ',new_dataset_save)
    protected,priv,unpriv,predicted,pref,unpref = get_labels(og_file,prot_var)
    new_dataset_save = new_dataset_save.rename(columns={"protected":protected,"predicted":predicted})
    new_dataset_save[protected]=np.where(new_dataset_save[protected]==1, priv, unpriv)
    new_dataset_save[predicted]=np.where(new_dataset_save[predicted]==1, pref, unpref) 
    print('AFTER: ',new_dataset_save)
    new_dataset_save.to_csv(file_name+'.csv')	
    return 

#################################################### MAIN ######################################################
if __name__ == "__main__":
    ############################
    # Get Data + One Hot Encode
    ############################
    global gen_reg
    global alg
    warnings.filterwarnings("ignore")
    directory = sys.argv[1]
    prot_var = sys.argv[2]
    if sys.argv[3] == 'True':
        from acc_fair_label_gpu import predict_output, train_on_data, get_train_test
    else:
        from acc_fair_label import predict_output, train_on_data, get_train_test
    alpha = 1

    # Get all Files
    og_file  = get_og_file(directory)
    og_data,_,ohe = get_data(og_file, prot_var)
    og_data_og = pd.read_csv(og_file, header='infer').dropna().reset_index(drop=True) 
    protected, _, _, predicted, _, _ = get_labels(og_file, prot_var)

    total_selected = len(og_data.index)
    interval = total_selected/10 
    retrain  = '1'
    alg = 'RandomForest'

    for label in ['1','2','3','4','5','6']:
        for to_select in [int(total_selected)/2]:
            train(directory, prot_var, retrain, label, alpha, int(to_select), alg)
