import argparse
import os
import re

import numpy as np
import pandas as pd

from datetime import datetime
from automsi import ae_preprocessing, ae_vae, ae_images, ae_recover, ae_utils



SSIM_FEATURE_REF_ID = 1583

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad_path', type=str, help='Path to h5ad files')
    parser.add_argument('--ae_path', type=str, help='Path to autoencoder experiment')
    parser.add_argument('--rf_path', type=str, help='Path to RF only experiments')
    parser.add_argument('--results_path', type=str, help='Path to results')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived.')
    parser.add_argument('--scaler', type=str, help='Scaler to normalize spatial data.')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names to encode with the autoencoder, separated by ;')
    parser.add_argument('--conv_ae_prefix', type=str, default="cor_info_ones_feature_", help='Prefix name of h5 files containing correlation information, only used if mode == conv_ae.')
    parser.add_argument('--latent_feature_id', type=ae_utils.split_int_at_semicolon, help='Id of latent features to recover information. Only the feature id matching with the suffix information is evaluated per run, only used if mode == conv_ae')
    parser.add_argument('--rf_cutoff', type=int, default=4, help='Cutoff for feature importance based on highest ranked feature score, e.g., 4th of score, only used if mode == rf.')
    parser.add_argument('--latent_feature_cor', type=float, default=0.95, help='Minimium Spearman correlation coefficient required to define original feature to be associated with latent feature.')
    parser.add_argument('--mode', type=str, default="con_vae", help='Define mode of data (convae or rf).')


    return parser.parse_args()
   
    
    
def main():     
    ssims_sample = ae_recover.MetricCollector(SSIM_FEATURE_REF_ID).derive_ssim(msi.train.x.images, cor_features)
    print(len(ssims_sample))
    
    today = datetime.today().strftime('%Y-%m-%d')
    file = "_".join([args.experiment, today, args.mode, "ssims"])
    if args.mode == "rf":
        file = "_".join([args.experiment, today, args.mode, str(args.rf_cutoff), "ssims"])
        
    with pd.ExcelWriter(args.results_path + file + ".xlsx") as writer:  
        pd.DataFrame(ssims_sample, columns = cor_features, index = args.samples).to_excel(writer)
 

              
if __name__ == '__main__':
    args = parse_args()
    exp = ae_vae.ConvAEExperiment(args.experiment)

    adata, _, n_features = ae_preprocessing.read_msi_from_adata(args.h5ad_path + exp.file + ae_utils.H5AD_SUFFIX + ".h5ad")
    
    if args.mode == "rf":
        cor_features = ae_recover.FeatureCollector(args.rf_cutoff).get_by_score(args.rf_path + args.experiment + ".xlsx", ['mz_index', 'fi'], "fi")
        
        experiment = args.experiment + str(args.rf_cutoff)
    elif args.mode == "conv_ae": 
        match = re.search(r'(\d+)$', args.experiment)
        suffix = int(match.group(1)) if match else 0
        latent_feature_id = args.latent_feature_id[suffix-1]
        
        path = os.path.join(args.ae_path, args.experiment, "")
        cor, cor_features = ae_recover.summarize_features_correlating_with_latent_feature(path, args.conv_ae_prefix, n_features, 500, latent_feature_id, args.latent_feature_cor, np.greater)
    
    print("Mode: ", args.mode, ". Number of m/z values considered: ", str(len(cor_features)))
    
    samples = adata[adata.obs["batch"].isin(args.samples)]
    samples, _ = ae_preprocessing.normalize_train_test_using_scaler(samples, None,  args.ae_path + "weights/" + args.scaler, pd.DataFrame(), None)
    msi = ae_images.MSIDataset(exp.patch_size, n_features).build(samples, None, create_patches = None)

    
    main()
    