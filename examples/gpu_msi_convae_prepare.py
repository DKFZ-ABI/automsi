import argparse
import sys

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
import pandas as pd
from automsi import *

import glob   
import re
import joblib 
import os
import h5py

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad_path', type=str, help='Path to spatial omics files')
    parser.add_argument('--msi_suffix', type=str, default="_labels", help='Suffix to specific msi_files that contain annotations, e.g., _labels')
    
    parser.add_argument('--ae_export', type=str, help='Path where scaler will be stored')
    parser.add_argument('--results_path', type=str, help='Path where patches will be stored')
    
    parser.add_argument('--train_samples', type=ae_utils.split_at_semicolon, help='Sample names to train autoencoder, separated by ;')
    parser.add_argument('--test_samples', type=ae_utils.split_at_semicolon, help='Sample names to test autoencoder, separated by ;')
    
    parser.add_argument('--h5ad_files', type=str, default="cal336o1t0_15", help='File name, h5ad file without file suffix')
        
   
    # autoencoder params
    parser.add_argument('--patch_size', type=int, default=3, help='Patch size of convolutional layers')
    parser.add_argument('--overlapping_patches', type=int, default=1, help='Stride for creating overlapping patches')
    parser.add_argument('--mode', type=str, default="unsupervised", help='Mode of autoencoder training, i.e., unsupervised or semi-supervised')
    parser.add_argument('--semi_label', type=str, default="FI", help='Label used for training, only used if mode = semi-supervised')
    
    return parser.parse_args()


def main(): 
    
    if args.overlapping_patches != 0:
        x_train, y_train = spt.train.get_non_empty_patches()
        x_train = np.asarray(x_train, dtype=np.float32) 
        y_train = np.asarray(y_train, dtype=np.float32) 

        x_test, y_test = spt.test.get_non_empty_patches()
        x_test =  np.asarray(x_test, dtype=np.float32) 
        y_test = np.asarray(y_test, dtype=np.float32)

        if args.mode == "unsupervised":
            datasets.write_tfr_x(x_train, args.results_path  + args.h5ad_files + "_train_" +  "".join(args.train_samples) + "_" + args.mode + "_" + suffix)
            datasets.write_tfr_x(x_test, args.results_path  + args.h5ad_files + "_test_" +  "".join(args.test_samples) + "_" + args.mode + "_" + suffix)
        elif args.mode == "semi-supervised":
            y_train = y_train[..., spt._images.im_label[args.semi_label]] 
            y_test = y_test[..., spt._images.im_label[args.semi_label]]
            #y_test = np.full(y_test.shape[0:3], 0.0) # if test has no labels
            
            datasets.write_tfr(x_train, y_train, args.results_path + args.h5ad_files + "_train_" + "".join(args.train_samples) + "_" + args.mode + "_" + suffix)
            datasets.write_tfr(x_test, y_test, args.results_path + args.h5ad_files + "_test_" + "".join(args.test_samples) + "_" + args.mode + "_" + suffix)
            
    else: # user for interpret.sh
        x_train = np.concatenate(spt.train.x.patches)
        x_train = np.asarray(x_train, dtype=np.float32) 
        
        index_split = np.cumsum([p.shape[0] for p in spt.train.x.patches])
    
        datasets.write_tfr_x_by_feature(x_train, args.results_path + args.h5ad_files + "_original_" + "".join(args.train_samples) + "_" + suffix)
        
        
if __name__ == '__main__':
    args = parse_args()
    naming = [args.patch_size] + [args.overlapping_patches]
    suffix = '_'.join(str(n) for n in naming)
    
    adata, _, n_features = ae_preprocessing.read_msi_from_adata(args.h5ad_path + args.h5ad_files + "_labels.h5ad")
    print(adata)
    
    train = adata[adata.obs["batch"].isin(args.train_samples)]
    test = adata[adata.obs["batch"].isin(args.test_samples)]
    
    scaled_samples = '_'.join(args.train_samples) 
    scaler = args.ae_export + "weights/" + args.h5ad_files + "_" + scaled_samples + "_" +  str(train.n_vars)
    train, test = ae_preprocessing.min_max_normalize_train_test(train, test, scaler)
    
    im_label =  {'FI': 0}
    spt = ae_images.MSIDataset(args.patch_size, adata.n_vars).build(train, test)
    if args.overlapping_patches != 0:
        spt.overlapping_patches(args.overlapping_patches, on_test = False)
    
    write_weights = True
    
    main()