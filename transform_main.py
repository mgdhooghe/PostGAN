import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_data(input_file, og_file, protected, privileged, predicted, preferred):
    og_data = pd.read_csv(og_file, header='infer')
    data = pd.read_csv(input_file, header='infer')
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
