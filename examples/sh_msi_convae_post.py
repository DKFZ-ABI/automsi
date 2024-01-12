import argparse
import os

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
import pandas as pd

from automsi import ae_preprocessing, ae_vae, ae_images, ae_utils, ae_plots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msi_import', type=str, help='Path to msi_files')
    parser.add_argument('--ae_path', type=str, help='Path to experiment')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names encoded with autoencoder, separated by ;')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived')
    parser.add_argument('--scaler', type=str, help='Scaler to normalize MSI data')
    return parser.parse_args()
   

def main(msi_experiment, msi, args):     
    p_flatten = np.concatenate(msi.train.x.patches)
        
    z = msi_experiment.conv_ae._encoder.predict(p_flatten) 
    z_patches, z_images = msi._patches.get_labeled_images_from_patches(msi.train.x.patches, z)
    path = os.path.join(args.ae_path, args.experiment, "latent")
    ae_plots.conv_generate_latent_plots_from_patches(z_images, 6, 5,  path + ".pdf")
                     
      
    
if __name__ == '__main__':
    args = parse_args()
    
    msi_experiment = ae_vae.ConvAEExperiment(args.experiment)
    adata, mz_values, n_features = ae_preprocessing.read_msi_from_adata(args.msi_import + msi_experiment.file + "_nlabels.h5ad")
    msi_experiment.build(n_features, args.ae_path, args.experiment)
    
    print("Loading patches from: ",  msi_experiment.file)
    print(adata.obs.iloc[0])
    
    msi_samples = adata[adata.obs["batch"].isin(args.samples)]
    scaler = args.ae_path + "weights/" + args.scaler
    msi_samples, _ = ae_preprocessing.normalize_train_test_using_scaler(msi_samples, None, scaler, pd.DataFrame(), None)
    
    obs_xy = ["xLocation", "yLocation"]
    msi = ae_images.SpatialDataset(msi_experiment.patch_size, n_features, None, {}, {}, obs_xy).build(msi_samples, None)
    
   

    plot_results = True
    main(msi_experiment, msi, args)
    