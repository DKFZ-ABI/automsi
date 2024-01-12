import argparse
import os
# to avoid nasty warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# reduce number of threads
os.environ['TF_NUM_INTEROP_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
import tensorflow as tf

import numpy as np
import pandas as pd

from automsi import ae_preprocessing, ae_vae, ae_images, ae_utils

from multiprocessing.pool import ThreadPool as Pool

import scipy.stats
import h5py
import skimage.transform as st
#from psutil import Process
import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad_path', type=str, help='Path to h5ad files')
    parser.add_argument('--ae_export', type=str, help='Path to experiment')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived.')
    parser.add_argument('--scaler', type=str, help='Scaler to normalize spatial data.')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names to encode with the autoencoder, separated by ;')
    # don't change order
    parser.add_argument('--backfill', type=str, default="ones", help='Noise values to fill features.')
    parser.add_argument('--latent_feature_id', type=int, help='Id of latent feature to recover information.')
    parser.add_argument('--feature_from', type=int, help='Starting feature id to analyze".')
    parser.add_argument('--feature_to', type=int, help='Final feature id to analyse (exclusive), maximal the number of total features.')
    # 
    parser.add_argument('--spatial_data', type=str, default="msi", help='Define spatial omics data (msi or spt).')
    parser.add_argument('--thread_pool', type=int, default=4, help='Number of threads to parallize code.')
    return parser.parse_args()



def encode_adjusted_patches(patches, feature_id):
    
    flatten_patches = np.concatenate(patches)
    
    if args.backfill == "ones":
        adj_patches = np.ones(flatten_patches.shape)
    else:
        adj_patches = np.zeros(flatten_patches.shape)
        
    adj_patches[..., feature_id] = flatten_patches[..., feature_id]
    adj_patches_tensor = tf.data.Dataset.from_tensor_slices(adj_patches).batch(adj_patches.shape[0])
    
    z = exp.conv_ae.encoder.predict(adj_patches_tensor, batch_size=1000) 
    _ = gc.collect()
        
    return z
    
    
# if this produces nans, backfill needs to be flipped
def compute_correlation_coefficient(z_images, x_images, feature_id):
    latent_im = z_images[...,args.latent_feature_id]
    org_im = x_images[..., feature_id]
    org_im = st.resize(org_im, latent_im.shape)
    
    # ignore background pixels for comparison
    org_im = np.where(org_im==0.0, np.nan, org_im)
    im_nans = np.isnan(org_im)
    
    cor = 0.0
    cm = scipy.stats.spearmanr(latent_im[~im_nans].flat, org_im[~im_nans].flat)
    if cm.pvalue < 0.05:
        cor = cm.correlation
    
    return cor
    
    
def compare_ion_image_against_latent(feature_id):
    
    z = encode_adjusted_patches(spatial.train.x.patches, feature_id)
    z_patches, z_images = spatial._patches.get_labeled_images_from_patches(spatial.train.x.patches, z)
        
    cor = np.zeros(len(z_images))
    np.seterr(divide='ignore', invalid='ignore')
    
    samples = len(spatial.train.x.patches)
    for s in range(samples):
        cor[s] = compute_correlation_coefficient(z_images[s], spatial.train.x.images[s], feature_id)
        
    mean_cor = 0. if np.any(cor == 0.) else np.mean(cor)
    return mean_cor
    


def main(args):     
    
    with Pool(args.thread_pool) as pool:
        cor = pool.map(compare_ion_image_against_latent, list(range(args.feature_from, args.feature_to)))
    print(cor[0])
    
    file = ["cor_info"] + list(vars(args).values())[5:9]
    h5f = h5py.File(os.path.join(args.ae_export, args.experiment, '_'.join(str(n) for n in file)  + ".h5") , 'w')
    h5f.create_dataset("cor", data=cor)
    h5f.close()
    
      
    
if __name__ == '__main__':
    args = parse_args()
    exp = ae_vae.ConvAEExperiment(args.experiment)
    
    if args.spatial_data == "msi":
        adata, _, n_features = ae_preprocessing.read_msi_from_adata(args.h5ad_path + exp.file + ae_utils.H5AD_SUFFIX + ".h5ad")
        spatial = ae_images.MSIDataset(exp.patch_size, n_features, im_label = {'FI': 0}, obs_label = {"FI" : "fi"})
        
    spt_samples = adata[adata.obs["batch"].isin(args.samples)]
    scaler = args.ae_export + "weights/" + args.scaler
    spt_samples, _ = ae_preprocessing.normalize_train_test_using_scaler(spt_samples, None, scaler, pd.DataFrame(), None)
    spatial.build(spt_samples, None, create_patches = True)
    exp.build(n_features, args.ae_export, args.experiment)
        
    plot_results = True
    
    main(args)
    
