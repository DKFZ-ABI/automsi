import argparse
import os

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad

from automsi import ae_preprocessing, ae_vae, ae_images, ae_utils

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp


LABEL = "FI"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad_path', type=str, help='Path to h5ad files')
    parser.add_argument('--ae_export', type=str, help='Path to experiment')
    parser.add_argument('--path_results', type=str, help='Path to export results')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived.')
    parser.add_argument('--scaler', type=str, help='Scaler to normalize spatial data.')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names to encode with the autoencoder, separated by ;')
    parser.add_argument('--feature_ids', type=ae_utils.split_int_at_semicolon, help='Ids of features to visualize.')
    parser.add_argument('--mode', type=str, help='Type of feature to visualize (ion, latent, reconstructed, hypxoa).')
    parser.add_argument('--file_suffix', type=str, default=".pdf", help='How to store visualization (pdf, png)".')
    return parser.parse_args()


def plot_multiple_generalized(image, other_image, title, file_name, feature_id):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(title, y=0.70)
    
    for i, _ in enumerate(axs.flat):
        idx = i//2
        im = axs[0, idx].imshow(image[idx][...,feature_id])
        axs[0, idx].set_title(image.index[idx])
        im = axs[1, idx].imshow(other_image[idx][...,feature_id])
        axs[1, idx].set_title(other_image.index[idx])

    cbar_ax = fig.add_axes([1, 0.52, 0.02, 0.325])
    fig.colorbar(im,  cax=cbar_ax)
    fig.tight_layout(rect=[0, 0.03, 1, 1.4])
    plt.savefig(args.path_results + args.experiment + file_name + str(feature_id) + ''.join(args.samples) + args.file_suffix, bbox_inches = "tight")  
    
    
    
def plot_multiple_with_overlay(image, overlay_label, label, title, file_name):
    sample_names = image.index.sort_values() 
    fig, axs = plt.subplots(1, len(sample_names))
    fig.suptitle(title)

    for i, name in enumerate(sample_names):
        if overlay_label is not None:
            axs[i].imshow(image[name][..., overlay_label], alpha = 0.5, cmap='gray')
        else:
            axs[i].imshow(msi.train.x.images[name][..., 0], alpha = 0.5, cmap='gray')
        im = axs[i].imshow(image[name][..., label], alpha = 0.8, vmin = 0.2, cmap=None)
        axs[i].set_title(name)
    
    cbar_ax = fig.add_axes([1, 0.52, 0.02, 0.325])
    fig.colorbar(im,  cax=cbar_ax)
    fig.tight_layout(rect=[0, 0.03, 1, 1.4])
    plt.savefig(args.path_results + args.experiment + file_name + ''.join(args.samples) + args.file_suffix, bbox_inches = "tight")  
    
    
    
def plot_generalized(image, y, label, title, file_name, feature_id):
    plt.rc('font', size=30) 
    plt.rc('legend', fontsize=30)
    plt.rcParams.update({'font.size': 30})
    
    sample_names = image.index.sort_values()   
    fig, axs = plt.subplots(1, len(sample_names), figsize=(20, 20))
     
    for i, name in enumerate(sample_names):
        im = axs[i].imshow(image[name][...,feature_id])
        axs[i].set_title(name)
       
   
    if plot_title:
        fig.suptitle(title)
        cbar_ax = fig.add_axes([1, 0.64, 0.02, 0.225])
        fig.colorbar(im,  cax=cbar_ax)
        fig.tight_layout(rect=[0, 0.03, 1, 1.55])
    else:
        cbar_ax = fig.add_axes([1, 0.42, 0.02, 0.2])
        fig.colorbar(im,  cax=cbar_ax)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(args.path_results + args.experiment + file_name + str(feature_id) + ''.join(args.samples) + args.file_suffix, bbox_inches = "tight")  
    
    
def main():     
  
    if args.mode == "latent" or args.mode == "reconstructed":
        exp.build(n_features, args.ae_export, args.experiment)
        
        p_flatten = np.concatenate(msi.train.x.patches)
        z = exp.conv_ae._encoder.predict(p_flatten) 
        z_patches, z_images = msi._patches.get_labeled_images_from_patches(msi.train.x.patches, z)
        
        if args.mode == "latent":
            for feature_id in args.feature_ids:
                plot_generalized(z_images, msi.train.y.images, msi._images.im_label[LABEL], "Latent m/z, id="+ str(feature_id), "_latent_", feature_id)
                
        elif args.mode == "reconstructed":
            dec = exp.conv_ae._decoder.predict(z) 
            dec_patches, dec_images = msi._patches.get_labeled_images_from_patches(msi.train.x.patches, dec)
            file = "_original-reconstructed_"
            
            for feature_id in args.feature_ids:
                mz_str = f'{mz_values[feature_id]:.3f}'
                title = "Original (top) vs reconstructed (bottom) m/z, id=" + str(feature_id) + ",\n m/z value=" + mz_str
                plot_multiple_generalized(msi.train.x.images, dec_images, title, file, feature_id)
        
    elif args.mode == "ion":
        for feature_id in args.feature_ids:
            mz_str = f'{mz_values[feature_id]:.3f}'
            title = "Ion image, m/z id=" + str(feature_id) + ", m/z value=" + mz_str
            plot_generalized(msi.train.x.images, msi.train.y.images,  msi._images.im_label[LABEL], title, "_ion_", feature_id)
    
    elif args.mode == "hypoxia":
        title = "Hypoxia annotations"
        #plot_multiple_with_overlay(msi.train.y.images, msi._images.im_label['HE'], msi._images.im_label[LABEL], title, "_hypoxia_annotations")
        plot_multiple_with_overlay(msi.train.y.images, None, msi._images.im_label[LABEL], title, "_hypoxia_annotations")
      
    
if __name__ == '__main__':
    args = parse_args()
    print(type(args.feature_ids[0]))
    print(args.feature_ids)
    
    exp = ae_vae.ConvAEExperiment(args.experiment)
    
    adata, mz_values, n_features = ae_preprocessing.read_msi_from_adata(args.h5ad_path + exp.file + ae_utils.H5AD_SUFFIX + ".h5ad")
    samples = adata[adata.obs["batch"].isin(args.samples)]
    scaler = args.ae_export + "weights/" + args.scaler
    samples, _ = ae_preprocessing.normalize_train_test_using_scaler(samples, None, scaler, pd.DataFrame(), None)
    
    msi = ae_images.MSIDataset(exp.patch_size, n_features, im_label = {'FI': 0}, obs_label = {"FI" : "fi"}).build(samples, None)

    plot_title = True
    
    main()
    