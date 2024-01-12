import os
import argparse

# reduce number of threads
os.environ['TF_NUM_INTEROP_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from automsi import ae_preprocessing, ae_vae, ae_images, ae_utils, ae_plots


import matplotlib.pyplot as plt

import joblib 
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msi_import', type=str, help='Path to msi_files')
    parser.add_argument('--ae_export', type=str, help='Path where results directory will be created')
    parser.add_argument('--train_samples', type=ae_utils.split_at_semicolon, help='Sample names to train autoencoder, separated by ;')
    parser.add_argument('--test_samples', type=ae_utils.split_at_semicolon, help='Sample names to test autoencoder, separated by ;')
    
    parser.add_argument('--msi_suffix', type=str, default="_labels", help='Suffix to specific msi_files that contain annotations, e.g., _labels')
    parser.add_argument('--msi_files', type=str, default="cal336o1t0_15", help='MSI file name, h5ad file without file suffix')
   
    # autoencoder params
    parser.add_argument('--patch_size', type=int, default=3, help='Patch size of convolutional layers')
    parser.add_argument('--activation', type=str, default="sigmoid", help='Last activation function')
    parser.add_argument('--weight', type=int, default=0, help='DEPRECATED, will be removed')
    parser.add_argument('--kernel', type=int, default=2, help='Kernel size of convolutional layers')
    parser.add_argument('--stride', type=float, default=1, help='Stride of convolutional layers')
    parser.add_argument('--conv_filter_max', type=int, default=1024, help='Node size of first hidden layer')
    parser.add_argument('--conv_filter_min', type=int, default=64, help='Node size of last hidden layer')
    parser.add_argument('--conv_filter_step', type=int, default=4, help='Step size to define number of hidden layers between max and min, consider 1024 | 512 | 256 | 128 | 64 | 32 | 16 | 8')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train autoencoder')
    
    parser.add_argument('--suffix', type=str, help='Suffix, e.g., to enumerate autoencoder runs (_001 to _010)')
    parser.add_argument('--mode', type=str, default="unsupervised", help='Mode of autoencoder training, i.e., unsupervised or semi-supervised')
    parser.add_argument('--semi_threshold', type=float, default=0.6, help='Cutoff value for labeled pixels, only used if mode = semi-supervised')
    parser.add_argument('--semi_label', type=str, default="FI", help='Label used for training, only used if mode = semi-supervised')
    return parser.parse_args()



def plot_history(history, key: str, max_loss_scale: int, max_kl_scale: int, plot_val: bool):
    ae_plots.plot_history(history, key, [0, max_loss_scale], plot_val, path_prefix + "/")
 
                
        
def main(args): 
    x_train, y_train = msi.train.get_non_empty_patches()
    x_test, y_test = msi.test.get_non_empty_patches()
    
    enc_conv_params, dec_conv_params = ae_vae.ConvParams.build_power2_3x1_conv_layers(args.conv_filter_max, args.conv_filter_min, args.conv_filter_step, args.kernel, args.stride, [0])
    [print(params) for params in enc_conv_params]
    [print(params) for params in dec_conv_params]
    
    conv_ae = ae_vae.ConvAE(n_features, args.patch_size, enc_conv_params, dec_conv_params, args.activation).build() 

    learning_rate = 1e-4
    print("Adam: " + str(learning_rate))
    print("Shape of training data: " + str(x_train.shape))
    print("Shape of test data: " + str(x_test.shape))
    
    if args.mode == "semi-supervised":
        ae = ae_vae.SemiSupervisedAETrainer(conv_ae.encoder, conv_ae.decoder, args.weight, args.semi_threshold)  
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        
        y_train = y_train[..., msi._images.im_label[args.semi_label]]
        y_test = np.full(y_test.shape[0:3], 0.0) # test data has no annotations
        
        history = ae.fit(x_train, y_train,
                         epochs = args.epochs, batch_size = 64, shuffle = True, verbose=0,
                         callbacks=(EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)),
                         validation_data = (x_test, y_test),
                     )
        plot_history(history, "loss", 2000, 100, True)
    elif args.mode == "unsupervised":
        ae = ae_vae.WeightedAETrainer(conv_ae.encoder, conv_ae.decoder, args.weight) 
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        
        history = ae.fit(x_train,
                         epochs = args.epochs, batch_size = 64, shuffle = True, verbose=0,
                         callbacks=(EarlyStopping(monitor="val_reconstruction_loss", patience=30, restore_best_weights=True)),
                         validation_data = (x_test, None)
                     )
        plot_history(history, "reconstruction_loss", 500, 100, True)
        
    
    if write_weights:
        ae.encoder.save_weights(args.ae_export + "weights/" + prefix + '_encoder_model_weights.h5')
        ae.decoder.save_weights(args.ae_export + "weights/" + prefix + '_decoder_model_weights.h5')


        
if __name__ == '__main__':
    args = parse_args()
    samples_no = "train_" + str(len(args.train_samples)) + "_test_" + str(len(args.test_samples))
    naming = list(vars(args).values())[5:15] + [samples_no] + [args.mode] + [args.suffix] 
    prefix = '_'.join(str(n) for n in naming)
    path_prefix = os.path.join(args.ae_export, prefix)
    
    print("Creating directory for: " + prefix)
    os.mkdir(path_prefix)
        
    adata, mz_values, n_features = ae_preprocessing.read_msi_from_adata(args.msi_import + args.msi_files + args.msi_suffix + ".h5ad")
    print("Loading patches from: " + args.msi_files + args.msi_suffix)
    print(adata.obs.iloc[0])
    
    train = adata[adata.obs["batch"].isin(args.train_samples)]
    test = adata[adata.obs["batch"].isin(args.test_samples)]
    scaled_samples = '_'.join(args.train_samples) 
    scaler = args.ae_export + "weights/" + args.msi_files + "_" + scaled_samples + "_" +  str(n_features)
    train, test = ae_preprocessing.min_max_normalize_train_test(train, test, scaler)
    
    msi = ae_images.MSIDataset(args.patch_size, n_features, im_label = {'FI': 0}, obs_label = {"FI" : "fi"}).build(train, test).overlapping_patches(args.patch_size - 1)
       
    write_weights = True
    
    main(args)