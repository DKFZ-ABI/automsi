import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# reduce number of threads
#os.environ['TF_NUM_INTEROP_THREADS'] = '1'
#os.environ['TF_NUM_INTRAOP_THREADS'] = '1'


import argparse
import sys
import re
import joblib 


import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print(tf.version.VERSION)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
print(tf.config.threading.get_inter_op_parallelism_threads())
print(tf.config.threading.get_intra_op_parallelism_threads())
gpus = tf.config.experimental.list_physical_devices('GPU')

# Enable memory growth for each GPU
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
from tensorflow.keras.callbacks import EarlyStopping

from automsi import ae_vae, ae_utils, ae_plots, datasets

import matplotlib.pyplot as plt


BATCH_SIZE = 32
BUFFER_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_path', type=str, help='Path to spatial omics files')
    parser.add_argument('--h5ad_files', type=str, help='To track back original h5 file')
    parser.add_argument('--ae_export', type=str, help='Path where results directory will be created')
    parser.add_argument('--train_samples', type=ae_utils.split_at_semicolon, help='Sample names to train autoencoder, separated by ;')
    parser.add_argument('--test_samples', type=ae_utils.split_at_semicolon, help='Sample names to test autoencoder, separated by ;')
    parser.add_argument('--n_features', type=int, default=18735, help='Number of total features')

    # autoencoder params
    parser.add_argument('--patch_size', type=int, default=3, help='Patch size of convolutional layers')
    parser.add_argument('--overlapping_patches', type=int, default=1, help='Stride for creating overlapping patches')
    parser.add_argument('--activation', type=str, default="sigmoid", help='Last activation function')
    parser.add_argument('--weight', type=int, default=0, help='Weight non-background pixels.')
    parser.add_argument('--kernel', type=int, default=2, help='Kernel size of convolutional layers')
    parser.add_argument('--stride', type=int, default=1, help='Stride of convolutional layers')
    parser.add_argument('--conv_filter_max', type=int, default=1024, help='Node size of first hidden layer')
    parser.add_argument('--conv_filter_min', type=int, default=64, help='Node size of last hidden layer')
    parser.add_argument('--conv_filter_step', type=int, default=4, help='Step size to define number of hidden layers between max and min, consider 1024 | 512 | 256 | 128 | 64 | 32 | 16 | 8')
    parser.add_argument('--conv_filter_1x1_pos', type=int, default=0, help='Position of 1x1 kernels')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train autoencoder')
    
    parser.add_argument('--suffix', type=str, help='Suffix, e.g., to enumerate autoencoder runs (_001 to _010)')
    parser.add_argument('--mode', type=str, default="unsupervised", help='Mode of autoencoder training, i.e., unsupervised or semi-supervised')
    parser.add_argument('--semi_threshold', type=float, default=0.6, help='Cutoff value for labeled pixels, only used if mode = semi-supervised')
    return parser.parse_args()

    
def plot_history(history, key: str, max_loss_scale: int, max_kl_scale: int, plot_val: bool):
    ae_plots.plot_history(history, key, [0, max_loss_scale], plot_val, path_prefix + "/")
         
    
        
def main(): 
    enc_conv_params, dec_conv_params = ae_vae.AdjConvParams.build_power2_conv_layers_1x1_middle(args.conv_filter_max, args.conv_filter_min, args.conv_filter_step, args.kernel, args.stride, [args.conv_filter_1x1_pos])
    [print(params) for params in enc_conv_params]
    [print(params) for params in dec_conv_params]
    
    conv_ae = ae_vae.ConvAE(args.n_features, args.patch_size, enc_conv_params, dec_conv_params, args.activation).build() 
    #conv_ae = ae_vae.ConvVAE(args.n_features, args.conv_filter_min, args.patch_size, enc_conv_params, dec_conv_params, args.activation).build()
    
    learning_rate = 1e-4
    print("Adam: " + str(learning_rate))    
    if args.mode == "semi-supervised":
        train = (tf.data.TFRecordDataset(tfr_files_train)
             .map(datasets.parse_complete_tfr, num_parallel_calls=1)
             .batch(BATCH_SIZE)
             .prefetch(buffer_size=BUFFER_SIZE)
            )
        
        test = (tf.data.TFRecordDataset(tfr_files_test)
            .map(datasets.parse_complete_tfr, num_parallel_calls=1)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=BUFFER_SIZE)
           )
        
        ae = ae_vae.SemiSupervisedAETrainer(conv_ae.encoder, conv_ae.decoder, args.weight, args.semi_threshold)  # 0.4
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        
        history = ae.fit(train,
                         epochs = args.epochs, batch_size = BATCH_SIZE, shuffle = True, verbose=2,
                         callbacks=(EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)),
                         validation_data = test,
                     )
        plot_history(history, "loss", 500, 50, True)
    elif args.mode == "unsupervised":
        train = (tf.data.TFRecordDataset(tfr_files_train)
             .map(datasets.parse_tfr_x, num_parallel_calls=1)
             .batch(BATCH_SIZE)
             .prefetch(buffer_size=BUFFER_SIZE)
            )
        
        test = (tf.data.TFRecordDataset(tfr_files_test)
            .map(datasets.parse_tfr_x, num_parallel_calls=1)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=BUFFER_SIZE)
           )
        
        #ae = ae_vae.VAETrainer(conv_ae.encoder, conv_ae.decoder, 1) 
        ae = ae_vae.WeightedAETrainer(conv_ae.encoder, conv_ae.decoder, args.weight)  
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        
        history = ae.fit(train, 
                      epochs = args.epochs, batch_size = BATCH_SIZE, shuffle = True,   verbose=2,
                      callbacks=(EarlyStopping(monitor="val_reconstruction_loss", patience=args.epochs, restore_best_weights=True)),
                      validation_data = (test, None),
                     )
     
        plot_history(history, "reconstruction_loss", 200, 20, True)
   
    
    if write_weights:
        ae.encoder.save_weights(args.ae_export + "weights/" + prefix + '_encoder_model_weights.h5')
        ae.decoder.save_weights(args.ae_export + "weights/" + prefix + '_decoder_model_weights.h5')


        
if __name__ == '__main__':
    args = parse_args()
    samples_no = "train_" + str(len(args.train_samples)) + "_test_" + str(len(args.test_samples))
    naming = [args.h5ad_files] + list(vars(args).values())[6:17] + [samples_no] + [args.mode] + [args.suffix] 
    
    prefix = '_'.join(str(n) for n in naming)
    path_prefix = os.path.join(args.ae_export, prefix)
    print("Creating directory for: " + prefix)
    os.mkdir(path_prefix)
    
    train_samples = '_'.join(str(n) for n in [args.h5ad_files, "train", "".join(args.train_samples), args.mode, args.patch_size, args.overlapping_patches])
    test_samples = '_'.join(str(n) for n in [args.h5ad_files, "test", "".join(args.test_samples), args.mode, args.patch_size, args.overlapping_patches])
    tfr_files_train =  [args.patch_path + train_samples + ".tfrecords"]
    tfr_files_test =  [args.patch_path + test_samples + ".tfrecords"]
    
    write_weights = True
    
    main()