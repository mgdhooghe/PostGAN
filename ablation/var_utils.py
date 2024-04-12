import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def train_on_data(data):
    reg = RandomForestClassifier(n_estimators=500)
    x_train = data[data.columns[:-1]].astype(float)
    y_train = data[data.columns[-1]].astype(float)
    
    if 'fnlwgt' in data.columns:
            weight = x_train['fnlwgt']
            x_train_with_weight = x_train.copy(deep=True)
            x_train = x_train.drop(columns=['fnlwgt'])
    else:
            weight = None
    
    try:
           #reg.fit(x_train,y_train, sample_weight = weight)
           reg.fit(x_train,y_train)
    except Exception as e:
            print("FAILED TRAINING: ",e)
            reg = -1
    
    if 'fnlwgt' in data.columns:
            x_train = x_train_with_weight
    
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

def get_labels(input_file, protected_var):
    #print(input_file)
    #print(protected_var)
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
        if protected_var=='age':
            protected='Age'
            privileged='1'
            unprivileged='0'
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

