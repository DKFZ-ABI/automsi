from sklearn import preprocessing
import pandas as pd
import numpy as np
import anndata as ad
import joblib

def normalize_with_obs(scale_fu, data: ad.AnnData, obs: pd.Series):
    data_df = data.to_df()
    norm = pd.DataFrame(scale_fu(data_df.values), columns=data_df.columns, index=data_df.index)
    return ad.AnnData(norm, obs=obs)

def normalize_obs(scale_fu, obs: pd.Series, label: str):
    # necessary if we normalize only one dimension
    data = np.array(obs[label]).reshape(-1, 1)
    obs[label + "_norm"] = scale_fu(data)
    return obs
    

def normalize(scale_fu, data: ad.AnnData):
    data_df = data.to_df()
    norm = pd.DataFrame(scale_fu(data_df.values), columns=data_df.columns, index=data_df.index)
    return ad.AnnData(norm, obs=data.obs)
    

    
def normalize_train_test_using_scaler(train : ad.AnnData, test : ad.AnnData, path_prefix : str, train_obs : pd.DataFrame, test_obs : pd.DataFrame) -> (ad.AnnData, ad.AnnData):
    if train_obs.empty:
        train_obs = train.obs.copy()
    if test is not None and test_obs.empty:
        test_obs = test.obs.copy()
    
    scaler = joblib.load(path_prefix + "_scaler.gz")
    train = normalize_with_obs(scaler.transform, train, train_obs)
    if test is not None:
        test = normalize_with_obs(scaler.transform, test, test_obs)
    
    return train, test
        
    
def min_max_normalize_train_test(train: ad.AnnData, test: ad.AnnData,
                                 path_prefix: str = "") -> (ad.AnnData, ad.AnnData):
        
    min_max_scaler = preprocessing.MinMaxScaler()
    train = normalize(min_max_scaler.fit_transform, train)
    if test is not None:
        test = normalize(min_max_scaler.transform, test)
    
    if path_prefix != "":  joblib.dump(min_max_scaler, path_prefix + "_scaler.gz") 

    return train, test


def min_max_normalize_labels(train_obs: pd.DataFrame, test_obs: pd.DataFrame, scale_label: str,
                             path_prefix: str = "") -> (pd.DataFrame, pd.DataFrame):
    label_min_max_scaler = preprocessing.MinMaxScaler()
    train_obs = normalize_obs(label_min_max_scaler.fit_transform, train_obs, scale_label)
    if test_obs is not None:
        test_obs = normalize_obs(label_min_max_scaler.transform, test_obs, scale_label)
    if path_prefix != "": joblib.dump(label_min_max_scaler, path_prefix + "_" + scale_label + "_scaler.gz")
    
    return train_obs, test_obs
    

    

def read_msi_from_adata(path: str) -> (ad.AnnData, pd.Index, int): 
    adata = ad.read(path)
    
    mz_values = pd.to_numeric(adata.to_df().columns, errors='coerce')
    n_features = len(mz_values)
    
    return adata, mz_values, n_features

