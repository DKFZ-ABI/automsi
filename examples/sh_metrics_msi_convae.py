import argparse
import os
import re
import glob

import numpy as np
import pandas as pd

from datetime import datetime
from automsi import ae_preprocessing, ae_vae, ae_images, ae_recover, ae_utils



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad_path', type=str, help='Path to h5ad files')
    parser.add_argument('--ae_path', type=str, help='Path to autoencoder experiment')
    parser.add_argument('--rf_path', type=str, help='Path to RF only experiments')
    parser.add_argument('--rf_file', type=str, help='File to RF results')
    parser.add_argument('--results_path', type=str, help='Path to results')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived.')
    parser.add_argument('--scaler', type=str, help='Scaler to normalize spatial data.')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names to encode with the autoencoder, separated by ;')
    parser.add_argument('--latent_feature_id', type=int, default=-1, help='Id of latent features to recover information. Only the feature id matching with the suffix information is evaluated per run, only used if mode == conv_ae. If -1 than not evaluated')
    parser.add_argument('--rf_cutoff', type=int, default=4, help='Cutoff for feature importance based on highest ranked feature score, e.g., 4th of score, only used if mode == rf.')
    parser.add_argument('--latent_feature_cor', type=float, default=0.95, help='Minimium Spearman correlation coefficient required to define original feature to be associated with latent feature.')
    parser.add_argument('--ssim_feature_ref_id', type=int, help='Reference feature ID to calculate SSIM.')
    parser.add_argument('--mode', type=str, default="con_vae", help='Define mode of data (convae or rf).')
    parser.add_argument('--overlapping_patches', type=int, default=1, help='Number of overlapping patches, only info.')


    return parser.parse_args()
   
def extract_params(path):
    matching_files = glob.glob(path + "*.h5")
    
    index = 0
    if args.latent_feature_id != -1:
        for idx, file_name in enumerate(matching_files):
            latent_pos = int(file_name.split('_')[-3])
            if latent_pos == args.latent_feature_id:
                index = idx
    print("Scanning...", matching_files[index])
    
    if matching_files:
        matching_files = os.path.basename(matching_files[index])
        file_name = matching_files.replace('.', '_')
        file_name = file_name.split('_')
        len_file_name = len(file_name)

        file_prefix = ("_").join(file_name[0:len_file_name-4])
        latent_id = int(file_name[len_file_name-4])
        step_size = int(file_name[len_file_name-2]) - int(file_name[len_file_name-3])

        return file_prefix + "_", latent_id, step_size
    
    return None, None, None



def get_correlated_features():
    cor_features = np.array([])
    
    if args.mode == "rf":
        cor_features = ae_recover.FeatureCollector(args.rf_cutoff).get_by_score(args.rf_path + args.rf_file + ".xlsx",  ['mz_index', 'fi'], "fi")
        
    elif args.mode == "conv_ae":         
        path = os.path.join(args.ae_path, args.experiment, "")
        file_prefix, latent_feature_id, step_size = extract_params(path)
        
        if file_prefix is not None:
            cor, cor_features = ae_recover.summarize_features_correlating_with_latent_feature(path, file_prefix, n_features, step_size, latent_feature_id, args.latent_feature_cor, np.greater)
    
    print("Mode: ", args.mode, ". Number of m/z values considered: ", str(len(cor_features)))
    return cor_features

    
    
def main():     
    cor_features = get_correlated_features()
    
    ssims_sample = ae_recover.MetricCollector(args.ssim_feature_ref_id).derive_ssim(msi.train.x.images, cor_features)
    print(len(ssims_sample))
    
    today = datetime.today().strftime('%Y-%m-%d')
    file = "_".join([args.experiment, str(args.latent_feature_id), today, args.mode, "ssims"])
    if args.mode == "rf":
        file = "_".join([args.experiment, today, args.mode, str(args.rf_cutoff), str(args.overlapping_patches), "ssims"])
        
    with pd.ExcelWriter(args.results_path + file + ".xlsx") as writer:  
        pd.DataFrame(ssims_sample, columns = cor_features, index = args.samples).to_excel(writer)
 

              
if __name__ == '__main__':
    args = parse_args()
    exp = ae_vae.ConvAEExperiment(args.experiment)
    adata, _, n_features = ae_preprocessing.read_msi_from_adata(args.h5ad_path + exp.file + ae_utils.H5AD_SUFFIX + ".h5ad")
    samples = adata[adata.obs["batch"].isin(args.samples)]
    samples, _ = ae_preprocessing.normalize_train_test_using_scaler(samples, None,  args.ae_path + "weights/" + args.scaler, pd.DataFrame(), None)
    msi = ae_images.MSIDataset(exp.patch_size, n_features).build(samples, None, create_patches = None)

    
    main()
    