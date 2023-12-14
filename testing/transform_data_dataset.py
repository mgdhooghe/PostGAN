import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_labels(input_file, protected_var):
    print(input_file)
    print(protected_var)
    if 'adult' in input_file:
        if protected_var=='race':
            protected = 'race'
            privileged = 'White'
            unprivileged = 'Other'
        if protected_var=='gender':
            protected = 'sex'
            privileged = 'Male'
            unprivileged = 'Female'
        predicted = 'income'
        preferred = '>50K'
        unpreferred = '<=50K'
    elif 'preproc_compas' in input_file or 'PROPUBLICA' in input_file or 'propublica' in input_file or 'compas' in input_file:
        if protected_var=='race':
            protected='race'
            privileged='Caucasian'
            unprivileged = 'Other'
            #privileged='1.0'
            #unprivileged='0.0'
        if protected_var=='gender':
            protected='sex'
            privileged='Male'
            unprivileged = 'Female'
        predicted='two_year_recid'
        preferred='0'
        unpreferred='1'
    elif 'census' in input_file:
        if protected_var=='race':
            protected='RACE'
            privileged='White'
            unprivileged = 'Other'
        if protected_var=='gender':
            protected='SEX'
            privileged='Male'
            unprivileged = 'Female'
        predicted='INCOME_50K'
        preferred='50000+.'
        unpreferred='-50000.'
    elif 'german' in input_file:
        if protected_var=='gender':
            protected='gender'
            privileged='male'
            unprivileged='female'
        predicted='labels'
        preferred='1'
        unpreferred='0'
    elif 'bank' in input_file:
        print('inbank')
        if protected_var=='marital':
            protected='marital'
            privileged='married'
            unprivileged='single'
        if protected_var=='age':
            protected='age'
            privileged='1'
            unprivileged='0'
        predicted='y'
        preferred='yes'
        unpreferred='no'
    elif 'medical' in input_file:
        if protected_var=='race':
            protected='RACE'
            privileged='1'
            unprivileged='0'
        predicted='UTILIZATION'
        preferred='1'
        unpreferred='0'
    return protected, privileged, unprivileged, predicted, preferred, unpreferred

def get_og_file(input_file, test=False, valid=False, all=False):
    if 'adult' in input_file:
        og_file = 'adult/ADULT-SPLIT-TRAIN-60.csv'
        og_test = 'adult/ADULT-SPLIT-TEST-20.csv'
        og_valid = 'adult/ADULT-SPLIT-VALID-20.csv'
        og_all_test = 'adult/ADULT-SPLIT-ALL-40.csv'
        ''' 
        og_file = 'adult/ADULT-SPLIT-TRAIN-60-noweight.csv'
        og_test = 'adult/ADULT-SPLIT-TEST-20-noweight.csv'
        og_valid = 'adult/ADULT-SPLIT-VALID-20-noweight.csv'
        og_all_test = 'adult/ADULT-SPLIT-ALL-40-noweight.csv'
        ''' 
    elif ('propublica' in input_file) or ('compas' in input_file):
        og_test = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TEST-20.csv'
        og_valid = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-VALID-20.csv'
        og_file = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TRAIN-60.csv'
        og_all_test = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-ALL-40.csv'
    elif 'census' in input_file:
        og_file = 'census/CENSUS-SPLIT-TRAIN.csv'
        og_test = 'census/CENSUS-SPLIT-TEST.csv'
    elif 'german' in input_file:
        og_file = 'german/GERMAN-SPLIT-TRAIN-60.csv'
        og_test = 'german/GERMAN-SPLIT-TEST-20.csv'
        og_valid = 'german/GERMAN-SPLIT-VALID-20.csv'
        og_all_test = 'german/GERMAN-SPLIT-ALL-40.csv'
    elif 'bank' in input_file:
        '''
        og_file = 'bank/bank-full-SPLIT-TRAIN-60.csv'
        og_test = 'bank/bank-full-SPLIT-TEST-20.csv'
        og_valid = 'bank/bank-full-SPLIT-VALID-20.csv'
        '''
        og_file = 'bank/bank-full-SPLIT-TRAIN-60-age.csv'
        og_test = 'bank/bank-full-SPLIT-TEST-20-age.csv'
        og_valid = 'bank/bank-full-SPLIT-VALID-20-age.csv'
        og_all_test = 'bank/bank-full-SPLIT-ALL-40-age.csv'
    elif 'medical' in input_file:
        og_file = 'medical/meps21-SPLIT-TRAIN-60.csv'
        og_test = 'medical/meps21-SPLIT-TEST-20.csv'
        og_valid = 'medical/meps21-SPLIT-VALID-20.csv'
        og_all_test = 'medical/meps21-SPLIT-ALL-40.csv'

    if test == True and valid == True and all == True:
        return og_file, og_test, og_valid, og_all_test
    elif test == True and valid == True:
        return og_file, og_test, og_valid
    elif test == True and all == True:
        return og_file, og_test, og_all_test
    elif valid == True and all == True:
        return og_file, og_valid, og_all_test
    elif test==True:
        return og_file, og_test
    elif valid==True:
        return og_file, og_valid
    elif all==True:
        return og_file, og_all_test
    else:
        return og_file


def get_data(input_file,protected_var,use=None):
    protected, privileged, unprivileged, predicted, preferred, unpreferred = get_labels(input_file, protected_var)
    og_file = get_og_file(input_file)
    og_data = pd.read_csv(og_file, header='infer')
    data = pd.read_csv(input_file, header='infer')
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    print(data)
    if protected not in data.columns:
        protected = 'protected'
    if predicted not in data.columns:
        predicted = 'predicted'
    data = data.dropna().reset_index(drop=True)
    read_data = data.copy()
    for col in read_data.columns:
        if data[col].dtypes in ['O',str]:
            #read_data[col] = read_data[col].apply(lambda x: x.str.strip())
            read_data[col] = read_data[col].str.strip()

    ### PROTECTED
    if data[protected].dtypes in ['O',str]:
        protected_data = pd.DataFrame(np.where(data[protected].str.strip() == privileged, 1, 0))
    elif data[protected].dtypes in [float]:
        protected_data = pd.DataFrame(np.where(data[protected].astype(int).astype(str) == privileged, 1, 0))
    else: 
        protected_data = pd.DataFrame(np.where(data[protected].astype(str) == privileged, 1, 0))
    protected_data.columns = [protected]
    data = data.drop([protected],axis=1)


    ### PREDICTED
    if data[predicted].dtypes in ['O',str]:
        predicted_data = pd.DataFrame(np.where(data[predicted].str.strip() == preferred, 1, 0))
    elif data[predicted].dtypes in [float]:
        predicted_data = pd.DataFrame(np.where(data[predicted].astype(int).astype(str) == preferred, 1, 0))
    else:
        predicted_data = pd.DataFrame(np.where(data[predicted].astype(str) == preferred, 1, 0))
    predicted_data.columns = [predicted]
    data = data.drop([predicted],axis=1)



    ## OHE
    ohe_columns = data.select_dtypes(['object']).columns
    print(ohe_columns)
    for col in ohe_columns:
        data[col] = data[col].str.strip()
    ohencoder = OneHotEncoder(handle_unknown='ignore').fit(data[ohe_columns])
    ohe_data = ohencoder.transform(data[ohe_columns]).toarray()
    ohe_data = pd.DataFrame(ohe_data)
    ohe_data.columns = ohencoder.get_feature_names_out()
    

    ### CONT
    cont_columns = data.drop(ohe_columns, axis=1)
    #################
    # Normalize Data
    #################
    for col in cont_columns.columns:
        #if abs(cont_columns[col]).max()!=0:
        #    cont_columns[col] = cont_columns[col].div(abs(cont_columns[col]).max())
        if abs(og_data[col]).max()!=0:
            cont_columns[col] = cont_columns[col].div(abs(og_data[col]).max())
    #################
    # Join Data
    ################# 
    transformed_data = cont_columns.join(ohe_data).join(protected_data).join(predicted_data)
    transformed_data = transformed_data.reset_index(drop=True)
    if use=='test':
        return transformed_data, read_data, protected

    return transformed_data, protected, ohencoder 


def return_data(data, ohe, og_data):
    ohe_length = len(ohe.get_feature_names_out())
    if ohe_length > 0:
        ohe_columns = ohe.feature_names_in_
    #############
    #GET PROTECTED/PREDICTED/CONTINUOUS DATA
    #############
    protect_predict = data.iloc[:,-2:]
    protect_predict.columns = ["protected","predicted"]
    #############
    #GET ONE HOT ENCODED DATA
    #############
    ohe_data = data.iloc[:,:-2]
    total = len(ohe_data.columns)
    ohe_data = ohe_data.iloc[:,total-ohe_length:]
    if ohe_length > 0:
        ohe_data = pd.DataFrame(ohe.inverse_transform(ohe_data.to_numpy()))
        ohe_data.columns = ohe_columns
    #############
    #RECOMBINE DATA
    #############
    cont_data = data.iloc[:,:total-ohe_length]
    if not isinstance(og_data, type(None)):
        for col in cont_data.columns:
            if abs(og_data[col]).max()!=0:
                cont_data[col] = cont_data[col]*abs(og_data[col]).max() 
                if isinstance(og_data[col][0], (int, np.integer)):
                    cont_data[col] = cont_data[col].astype(int)
    fin_data = cont_data.join(ohe_data).join(protect_predict)
    return fin_data#, protected
