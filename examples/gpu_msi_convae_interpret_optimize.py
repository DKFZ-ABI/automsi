import argparse
import os
# to avoid nasty warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow as tf
# reduce number of threads
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
print(tf.version.VERSION)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
import logging
tf.get_logger().setLevel(logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')

# Enable memory growth for each GPU
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
import numpy as np
import pandas as pd

from automsi.ae_vae import ConvAEExperiment, ConvVAEExperiment
from automsi.ae_images import Patches
from automsi import ae_utils, datasets
#from autospt import msi_extension, spt_datasets

import scipy.stats
import h5py
import skimage.transform as st
import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_path', type=str, help='Path to patch file')
    parser.add_argument('--ae_export', type=str, help='Path to experiment')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived.')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names to encode with the autoencoder, separated by ;')
    # don't change order
    parser.add_argument('--backfill', type=str, default="ones", help='Noise values to fill features.')
    parser.add_argument('--latent_feature_id', type=int, help='Id of latent feature to recover information.')
    parser.add_argument('--n_patches', type=int, help='Number of patches to process.')
    parser.add_argument('--index_split', type=ae_utils.split_int_at_semicolon, help='Denotes which patches belong to a sample.')
    parser.add_argument('--patches_split', type=ae_utils.split_int_at_semicolon, help='Denotes the number of patches in a row, only used for ConvVAE.')
    parser.add_argument('--n_features', type=int, default=18735, help='Number of total features')
    return parser.parse_args()


def update_tensor(feature_patches, dummy_tensor, feature_id):
    transposed_tensor = tf.transpose(feature_patches, perm=[1, 2, 3, 0])
    
    if feature_id == 0:
        updated_tensor = tf.concat([transposed_tensor, dummy_tensor[..., feature_id+1:]], axis=-1)
    elif feature_id == args.n_features-1:
        updated_tensor = tf.concat([dummy_tensor[..., :feature_id], transposed_tensor], axis=-1)
    else:
        updated_tensor = tf.concat([dummy_tensor[..., :feature_id], transposed_tensor, dummy_tensor[..., feature_id+1:]], axis=-1)
    
    return updated_tensor


    
# if this produces nans, backfill needs to be flipped
def compute_correlation_coefficient(latent_im, org_im):
    org_im = st.resize(org_im, latent_im.shape)
    
    # ignore background pixels for comparison
    org_im = np.where(org_im==0.0, np.nan, org_im)
    im_nans = np.isnan(org_im)
    
    cor = 0.0
    cm = scipy.stats.spearmanr(latent_im[~im_nans].flat, org_im[~im_nans].flat)
    if cm.pvalue < 0.05:
        cor = cm.correlation
    
    return cor
    
    
    
def main(args):     
    cor = np.zeros(args.n_features)
    
    for feature_id, batch in enumerate(data):
        # updated_tensor only contains information at position [...,feature_id]
        updated_tensor = update_tensor(batch, dummy_tensor, feature_id)
        z = exp.conv_ae.encoder.predict(updated_tensor, verbose = 0)
        _ = gc.collect()
        # rebuild image from latent feature of interest
        z_images = Patches.rebuild_images_from_indices(z, args.index_split, args.latent_feature_id)
        # vae
        #z_images = ae_utils.rebuild_flat_images(z[0], args.latent_feature_id, args.index_split, args.patches_split)
        # and compare it against its original 
        org_images = Patches.rebuild_images_from_indices(updated_tensor, args.index_split, feature_id)
        
        sample_cor = np.zeros(len(args.index_split))
        np.seterr(divide='ignore', invalid='ignore')

        samples = len(args.index_split)
        for s in range(samples):
            sample_cor[s] = compute_correlation_coefficient(z_images[s], org_images[s])

        mean_sample_cor = 0. if np.any(sample_cor == 0.) else np.mean(sample_cor)
        cor[feature_id] = mean_sample_cor 
        
    
    file = ["cor_info_complete"] + list(vars(args).values())[4:5+1]
    h5f = h5py.File(os.path.join(args.ae_export, args.experiment, '_'.join(str(n) for n in file)  + "_0_" + str(args.n_features) + "_TEST.h5") , 'w')
    h5f.create_dataset("cor", data=cor)
    h5f.close()
    
      
    
if __name__ == '__main__':
    args = parse_args()
    exp = ConvAEExperiment(args.experiment)
    #exp = ConvVAEExperiment(args.experiment)
    
    exp.build(args.n_features, args.ae_export, args.experiment)
    
    sample_names = "".join(args.samples)
    BATCH_SIZE = 1
    tfr_files_data =  [args.patches_path + exp.file + "_original_" + sample_names + "_" + str(exp.patch_size) + "_0.tfrecords"]
    
    if args.backfill == "zeros":
        dummy_tensor = tf.zeros(shape=(args.n_patches, exp.patch_size, exp.patch_size, args.n_features), dtype=tf.float32)
    else:
        dummy_tensor = tf.ones(shape=(args.n_patches, exp.patch_size, exp.patch_size, args.n_features), dtype=tf.float32)
        
    data = (tf.data.TFRecordDataset(tfr_files_data)
             .map(datasets.parse_tfr_x_by_feature, num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(BATCH_SIZE)
             .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )
      
    main(args)
    
