import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
import anndata as ad
import re
from typing import Final
import glob
import re
from automsi import ae_utils
INVALID_CLASS: Final = 99
    
def intensities_generator(p : ImzMLParser) -> np.ndarray:
    for i in range(len(p.coordinates)):
        _, unique_mz = np.unique(p.getspectrum(i)[0], return_index=True)
        full_spec = p.getspectrum(i)[1][unique_mz]
        yield full_spec        
        
def get_average_intensity_per_mz(frame) -> float:
    f = frame.copy()
    f[f==0.0] = np.nan
    avg = np.nanmedian(f, axis = 0)
    return avg

def get_raw_ms_and_scaling_factor(imzML : list) -> (list, list):

    ms = []
    scaling_factor = []
    for i in range(0, len(imzML)):
        p = ImzMLParser(imzML[i])
        print("Import" + imzML[i])
        
        num_spectra = len(p.mzLengths)
        mz_index = np.unique(np.concatenate([p.getspectrum(i)[0] for i in range(num_spectra)]))
        msi_frame = pd.DataFrame(intensities_generator(p), columns=mz_index)

        ms.append(msi_frame)
        scaling_factor.append(get_average_intensity_per_mz(msi_frame))
        
    return ms, scaling_factor


def add_annotations(obs : pd.DataFrame, file : str, name : str, sep = ";") -> pd.DataFrame:
    sample = re.search(ae_utils.SAMPLES, file).group(0)    
   
    annotations = pd.read_csv(file, sep = sep, header = None)
    xy_min = obs[["xLocation", "yLocation"]].min() 
    # this will create a dictionary for every pixel, (row, col): value
    anno_dict = annotations.transpose().stack().to_dict()
    # this will create for every mz pixel a spot to iterate over, (row, col)
    all_keys = list(zip(obs.xLocation - xy_min.xLocation, obs.yLocation - xy_min.yLocation)) 
    # mapping
    all_val = [anno_dict[x] for x in all_keys] 
   
    obs = obs.join(pd.DataFrame({name: all_val}))
    return obs


def normalize(imzML : list, global_scaling : bool, log_transform : bool) -> ad.AnnData:

    ms, average_intensity_per_mz = get_raw_ms_and_scaling_factor(imzML)
    df_average_intensity_per_mz = pd.DataFrame(average_intensity_per_mz, columns = ms[0].columns)
    global_scaling_factor = np.median(df_average_intensity_per_mz, axis = 0)
    scaling_factor = df_average_intensity_per_mz / global_scaling_factor
    
    df = []
    samples = []
    for i in range(len(imzML)):
        p = ImzMLParser(imzML[i])
        sample = re.search(ae_utils.SAMPLES, imzML[i]).group(0)  

        data = ms[i] / scaling_factor.loc[0] if global_scaling else ms[i] 
        data = np.log(data+10) if log_transform else data
        data = data.dropna(axis='columns')

        obs = pd.DataFrame(p.coordinates, columns=["xLocation", "yLocation", "zLocation"])
        obs = obs.drop(columns="zLocation")

        annData = ad.AnnData(data, obs=obs)
        df.append(annData)  
        samples.append(sample) 

    all_data = df[0].concatenate(df[1:len(imzML)], batch_categories = samples)

    
    if global_scaling:
        print("Average median intensities per sample:")
        print(average_intensity_per_mz)
        print("Applied the following scaling factors:")
        print(scaling_factor)
    
    return all_data


def annotate(all_data: ad.AnnData, he_path : str, fi_path: str, invalid_he_annotations: list, invalid_fi_annotations: list, sep: str):
    df = []
    samples = []
    
    for sample in all_data.obs.batch.unique():
        data = all_data[all_data.obs["batch"].isin([sample])]
        obs = data.obs.reset_index(drop=True) 
        
        if sample not in invalid_he_annotations:
            im_path = glob.glob(he_path + "*.txt") 
            im_index = [idx for idx, s in enumerate(im_path) if sample in s][0]
            obs = add_annotations(obs, im_path[im_index], "he", sep)

        if sample not in invalid_fi_annotations:
            im_path = glob.glob(fi_path + "*.txt") 
            im_index = [idx for idx, s in enumerate(im_path) if sample in s][0]
            obs = add_annotations(obs, im_path[im_index], "fi", sep)

        annData = ad.AnnData(data.to_df().reset_index(drop=True), obs=obs)
        df.append(annData)  
        samples.append(sample) 

    data_dict = {name: d for d, name in zip(df, samples)}
    annotated_data = ad.concat(data_dict, join="outer", label="batch", index_unique="-")
    
    return annotated_data
    