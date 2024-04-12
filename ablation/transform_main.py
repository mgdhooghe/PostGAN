import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

def get_data(input_file, og_file, protected = "protected", privileged = "priv", predicted = "predicted", preferred = "pref"):
    og_data = pd.read_csv(og_file, header='infer')
    data = pd.read_csv(input_file, header='infer')
    ### CHECK IF BANK
    if 'bank' in input_file or 'bank' in og_file:
        if protected == 'age':
            data['age'] = data['age'].astype(str)
            if data['age'].nunique() > 2:
                print("Updating Age")
                try:
                    data['age'] = np.where((data['age'].astype(int) < 25) | (data['age'].astype(int) > 60), 0, 1) 
                except Exception as e:
                    print('Number of Unique: ',data['age'].nunique())
                    print('Unique: ',data['age'].unique())
                    print('Exception: ',e)                    
    ### CHECK IF GERMAN 
    if 'german' in input_file or 'german' in og_file:
        if protected == 'Age':
            data['Age'] = data['Age'].astype(str)
            if data['Age'].nunique() > 2:
                print("Updating Age")
                try:
                    data['Age'] = np.where((data['Age'].astype(int) > 25), 1, 0) 
                except Exception as e:
                    print('Number of Unique: ',data['Age'].nunique())
                    print('Unique: ',data['Age'].unique())
                    print('Exception: ',e)                    

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
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
    return transformed_data, protected, ohencoder 

def get_data_plain(input_file, og_file):
    og_data = pd.read_csv(og_file, header='infer')
    data = pd.read_csv(input_file, header='infer')
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    data = data.dropna().reset_index(drop=True)

    ## OHE
    ohe_columns = data.select_dtypes(['object']).columns
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
    transformed_data = cont_columns.join(ohe_data)
    transformed_data = transformed_data.reset_index(drop=True)
    return transformed_data 

def return_data(data, ohe, og_file, prot="protected", pred="predicted", priv="priv", pref="unpref"):
    og_data = pd.read_csv(og_file, header='infer')
    ohe_length = len(ohe.get_feature_names_out())
    if ohe_length > 0:
        ohe_columns = ohe.feature_names_in_
    #############
    #GET PROTECTED/PREDICTED/CONTINUOUS DATA
    #############
    protect_predict = data.iloc[:,-2:]
    protect_predict.columns = [prot, pred]
    protect_predict[prot] = np.where(protect_predict[prot] == 1, priv, "unpriv")
    protect_predict[pred] = np.where(protect_predict[pred] == 1, pref, "unpref")
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
