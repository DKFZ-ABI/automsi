import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1
import pandas as pd
import numpy as np

import random
from abc import abstractmethod

from sklearn.model_selection import train_test_split
import gc

class AE(object):
    
    def __init__ (self, n_features : int):
        self.n_features = n_features
        self._encoder = None
        self._decoder = None
    
    @property
    def encoder(self):
        if self._encoder is None:
            raise Exception('Encoder is not build yet.')
        return self._encoder
    
    @property
    def decoder(self):
        if self._decoder is None:
            raise Exception('Decoder is not build yet.')
        return self._decoder
        
        
    @abstractmethod
    def build(self, plot_vae = False):
        pass
    
    
    def predict(self, samples, verbose = 0):
        p_flatten = np.concatenate(samples)
        encoded = self._encoder.predict(p_flatten, verbose=verbose) 
        latent_z = np.squeeze(encoded)
        decoded_data = self._decoder.predict(encoded, verbose=verbose) 

        return latent_z, decoded_data
    


class ConvAE(AE):
    
    def __init__ (self, n_features : int, patch_size : int, enc_conv_params : list, dec_conv_params : list, activation : str):
        super().__init__(n_features)
        self.patch_size = patch_size
        self.enc_conv_params = enc_conv_params
        self.dec_conv_params = dec_conv_params
        self.activation = activation
   
     
    def build(self, plot_vae = False):
        inputs = tf.keras.Input(shape = (self.patch_size, self.patch_size, self.n_features))
        
        h = inputs
        for params in self.enc_conv_params:
            h = self.build_conv2d_layers(params, h)
        encoder = tf.keras.Model(inputs, h, name = 'encoder')

        hdec = h
        for params in self.dec_conv_params:
             hdec = self.build_conv2dtranspose_layers(params, hdec)
     
        decoder_outputs = layers.Conv2DTranspose(self.n_features, (self.enc_conv_params[0].kernel, self.enc_conv_params[0].kernel), strides=self.enc_conv_params[0].strides, activation=self.activation, padding="valid")(hdec)
        decoder = tf.keras.Model(h, decoder_outputs, name = 'decoder')
        
        
        self._encoder = encoder
        self._decoder = decoder
        return self

    def build_conv2d_layers(self, param, inputs):
        h = layers.Conv2D(param.filters, (param.kernel, param.kernel), strides=param.strides, padding="valid")(inputs)
        h = layers.BatchNormalization()(h) # added
        h = layers.ReLU()(h) 
        return(h)    
    
    def build_conv2dtranspose_layers(self, param, inputs):
        hdec = layers.Conv2DTranspose(param.filters, (param.kernel, param.kernel), strides=param.strides, padding="valid")(inputs)
        hdec = layers.BatchNormalization()(hdec) # added
        hdec = layers.ReLU()(hdec) 
           
        return(hdec)
      
    

class CustomLoss():
    
    @staticmethod
    def reconstruction_base(sdiff):
        return tf.reduce_mean(
            tf.reduce_mean(sdiff, axis = (1,2)), 
        axis = 0) # reduce batch
    
    
    @staticmethod
    def post_norm(error, patch_size):
        patch_size = patch_size * patch_size
        error = error / tf.cast(patch_size, tf.float32) 
        return error
        
        
    @staticmethod
    def reconstruction_error(x, x_pred, weight):
        sdiff = tf.math.abs(x - x_pred)     
        sdiff = tf.multiply(sdiff, weight)
        
        reconstruction_err = CustomLoss.post_norm(CustomLoss.reconstruction_base(
           tf.reduce_sum(sdiff, axis=(3)) # reduce m/z values
        ), tf.shape(x)[1])
        
        return reconstruction_err
    
    
    @staticmethod
    def supervised_error(x, x_pred, y, threshold):
        # retains x.shape, e.g., (batch_size, 3, 3, 18375)
        sdiff = tf.math.abs(x - x_pred) # retains x.shape
        
        # retains y.shape, e.g., (batch_size, 3, 3)
        y_mask = tf.math.greater(y, threshold) 
        # down on pixel level, e.g., (matching_pixel, 18735)
        sdiff_weighted = tf.boolean_mask(sdiff, y_mask)
        
        is_shape_zero = tf.reduce_all(tf.equal(tf.shape(sdiff_weighted)[0], 0))

        def calculate_supervised_loss():
             return tf.reduce_sum(sdiff_weighted, axis=1)

        def return_empty_tensor():
            return tf.zeros(1, dtype=tf.float32)

        # down to single error per pixel, e.g., (error_per_pixel,)
        supervised_error = tf.cond(is_shape_zero, return_empty_tensor, calculate_supervised_loss)
        supervised_error = tf.reduce_mean(supervised_error)
        
        return supervised_error
    

    

class WeightedAETrainer(tf.keras.Model, CustomLoss):
    def __init__(self, encoder, decoder, weight, **kwargs):
        super(WeightedAETrainer, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.weight = weight
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
        ]

    def do_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            
        x = data
        
        # mask pixels which are zero
        zero_mask = tf.math.equal(x, 0.0)
        # inverse to get non-zero pixels
        non_zero_mask =  tf.logical_not(zero_mask)
        # set non-zero pixels to 1, and zero-pixel to 0
        non_zero_mask = tf.cast(non_zero_mask, tf.float32)
        # multiply by pixel weight, add one to essentially disable weighting in case of weight is set to 0
        pixel_weight = tf.math.add(tf.math.multiply(non_zero_mask, self.weight), 1)

        z = self.encoder(x)
        x_pred = self.decoder(z)

        reconstruction_loss = self.reconstruction_error(tf.cast(x, tf.float32), x_pred, pixel_weight)
        
        return x_pred, reconstruction_loss
                   
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            _, reconstruction_loss = self.do_step(data)
    
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }   

    
    def test_step(self, data):
        _, reconstruction_loss = self.do_step(data)
        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": reconstruction_loss,
            
        }




class SemiSupervisedAETrainer(WeightedAETrainer):
    def __init__(self, encoder, decoder, weight, supervised_threshold, **kwargs):
        super().__init__(encoder, decoder, weight, **kwargs)
        self.supervised_threshold = supervised_threshold
        self.supervised_loss_tracker = tf.keras.metrics.Mean(name="supervised_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.supervised_loss_tracker,
            self.total_loss_tracker,
        ]
    

    def do_step(self, data):
        x_pred, reconstruction_loss = super().do_step(data)
        
        x, y = data
        
        supervised_loss = self.supervised_error(tf.cast(x, tf.float32), x_pred, tf.cast(y, tf.float32), self.supervised_threshold)
        
        total_loss = reconstruction_loss + supervised_loss 
        return reconstruction_loss, supervised_loss, total_loss
                   
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction_loss, supervised_loss, total_loss = self.do_step(data)
    
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.supervised_loss_tracker.update_state(supervised_loss)
        self.total_loss_tracker.update_state(total_loss)
        
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "supervised_loss": self.supervised_loss_tracker.result(),
            "loss": self.total_loss_tracker.result(),
        }   

    
    def test_step(self, data):
        reconstruction_loss, supervised_loss, total_loss = self.do_step(data)
        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.supervised_loss_tracker.update_state(supervised_loss)
        self.total_loss_tracker.update_state(total_loss)
        
        return {
            "reconstruction_loss": reconstruction_loss,
            "supervised_loss": supervised_loss,
            "loss": total_loss,
            
        }
               

        
######## VAE        
    

class VAE(AE):
    
    def __init__ (self, n_features : int, latent_dim : int):
        super().__init__(n_features)
        self.latent_dim = latent_dim
    
    def sampling(self, args):
        z_mean, z_log_var  = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))  #    mean=0., stddev=0.1)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon 

    
    def predict(self, samples):
        p_flatten = np.concatenate(samples)
        encoded_data = self._encoder.predict(p_flatten) 
        latent_mean, latent_var, latent_z = encoded_data
        decoded_data = self._decoder.predict(latent_z) 
        return latent_z, decoded_data


class ConvVAE(VAE):
    
    def __init__ (self, n_features : int, latent_dim : int,
                 patch_size : int, conv_params : list, activation : str):
        super().__init__(n_features, latent_dim)
        self.patch_size = patch_size
        self.conv_params = conv_params
        self.activation = activation
   
     
    def build(self, plot_vae = False):
        divisor = np.sum([p.strides if p.strides > 1 else 0 for p in self.conv_params])
        divisor = 1 if divisor == 0 else divisor
        inputs = tf.keras.Input(shape = (self.patch_size, self.patch_size, self.n_features))
        h = self.build_conv2d_layers(self.conv_params[0], inputs)
        for params in self.conv_params[1:]:
            h = self.build_conv2d_layers(params, h)
        
        h = layers.Flatten()(h)
        
        h = layers.Dense(self.intermediate_dim)(h)
        h = layers.BatchNormalization()(h) 
        h = layers.ReLU()(h) 
        
        z_mean = layers.Dense(self.latent_dim, name = 'hidden_zmean_dense')(h)
        z_mean = layers.BatchNormalization(name = 'hidden_zmean_batch')(z_mean)
        z_log_var = layers.Dense(self.latent_dim, name = 'hidden_zlog_dense')(h)
        z_log_var = layers.BatchNormalization(name = 'hidden_zlog_batch')(z_log_var)
        z = layers.Lambda(self.sampling)([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name = 'encoder')
        
        if plot_vae:
            tf.keras.utils.plot_model(encoder, to_file='vae_encoder_conv.png', show_shapes = True)
            
        # decoder
        last_conv_layer = len(self.conv_params) - 1
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,), name = 'z_sampling')
        
        hdec = layers.Dense(int(self.patch_size/divisor) * int(self.patch_size/divisor) * 
                            self.conv_params[last_conv_layer].filters)(latent_inputs)
        hdec = layers.BatchNormalization()(hdec) 
        hdec = layers.ReLU()(hdec) 
        hdec = layers.Reshape((int(self.patch_size/divisor), int(self.patch_size/divisor), self.conv_params[last_conv_layer].filters))(hdec) ####
        
        reverse_params = self.conv_params[::-1] 
        for i, _ in enumerate(reverse_params):
            if i == 0:
                hdec = self.build_conv2dtranspose_layers(self.conv_params[last_conv_layer], hdec)
            else: 
                hdec = self.build_conv2dtranspose_layers_prev(reverse_params[i-1], reverse_params[i].filters, hdec)
        
         # rebuild to original feature space; probably kernel should be similar to first layer?
        latent_outputs = layers.Conv2DTranspose(self.n_features, (self.conv_params[0].kernel, self.conv_params[0].kernel), 
                                                 strides=self.conv_params[0].strides, activation=self.activation, padding="same")(hdec)
        
        decoder = tf.keras.Model(latent_inputs, latent_outputs, name = 'decoder')
        
        if plot_vae:
            tf.keras.utils.plot_model(decoder, to_file='vae_decoder_conv.png', show_shapes=True)   
        
        self._encoder = encoder
        self._decoder = decoder
        return self

    def build_conv2d_layers(self, param, inputs):
        h = layers.Conv2D(param.filters, (param.kernel, param.kernel), strides= param.strides, padding="same")(inputs)
        h = layers.BatchNormalization()(h) 
        h = layers.ReLU()(h) 
        return(h)
    
    def build_conv2dtranspose_layers(self, param, inputs):
        hdec = layers.Conv2DTranspose(param.filters, (param.kernel, param.kernel), strides= param.strides, padding="same")(inputs)
        hdec = layers.BatchNormalization()(hdec)
        hdec = layers.ReLU()(hdec) 
        return(hdec)
    
    def build_conv2dtranspose_layers_prev(self, prev_param, filters, inputs):
        hdec = layers.Conv2DTranspose(filters, (prev_param.kernel, prev_param.kernel), strides= prev_param.strides, padding="same")(inputs)
        hdec = layers.BatchNormalization()(hdec)
        hdec = layers.ReLU()(hdec) 
        return(hdec)

    

class CustomWeightedLoss(CustomLoss):
    
    
    @staticmethod
    def post_norm(error, patch_size, rec_coef):
        patch_size = patch_size * patch_size
        error = error / tf.cast(patch_size, tf.float32) 
        error = error * rec_coef
        return error
        
    # overwrites base class, and cannot be called (anyway static)
    @staticmethod
    def reconstruction_error(y, y_pred, rec_coef):
        sdiff = tf.math.abs(y - y_pred) 
        reconstruction_err = CustomLoss.post_norm(CustomLoss.reconstruction_base(
           tf.reduce_sum(sdiff, axis=(3)), # reduce m/z values
        ), tf.shape(y)[1], rec_coef)
        return reconstruction_err
    
        
    @staticmethod
    def reconstruction_error_per_mz(y, y_pred, rec_coef):
        sdiff = tf.math.abs(y - y_pred) 
        reconstruction_err = CustomLoss.post_norm(CustomLoss.reconstruction_base(
            sdiff # keep m/z values
        ), tf.shape(y)[1], rec_coef)
        return reconstruction_err
    
    


class VAETrainer(tf.keras.Model, CustomWeightedLoss):
    def __init__(self, encoder, decoder, kl_coef, rec_coef, **kwargs):
        super(VAETrainer, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_coef = kl_coef
        self.rec_coef = rec_coef
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    


    def do_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        data_nans = tf.math.is_nan(data)
        data = tf.where(data_nans, tf.zeros_like(data), data)
            
        z_mean, z_log_var, z = self.encoder(data)
        data_pred = self.decoder(z)
        reconstruction_loss = self.reconstruction_error(tf.cast(data, tf.float32), data_pred, self.rec_coef)
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.kl_coef * kl_loss 
        return reconstruction_loss, kl_loss, total_loss
                   
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction_loss, kl_loss, total_loss = self.do_step(data)
    
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }   

    
    def test_step(self, data):
        reconstruction_loss, kl_loss, total_loss = self.do_step(data)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
                    

        
##### Shared 
        
    
class UpdateNoiseCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, lower, upper, **kwargs):
        super(UpdateNoiseCallback, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper


    def on_epoch_begin(self, epoch, logs=None):
        gaussian_layer = self.model.get_layer('encoder').layers[2]
        gaussian_layer.stddev = random.uniform(self.lower, self.upper)
        print('updating sttdev in training')
        print(gaussian_layer.stddev)
        
        
        
        

class ConvParams():
    def __init__(self, filters, kernel, strides):
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        
    def __str__(self):
        return str(self.filters)+ "x" + str(self.kernel) +"x" + str(self.strides)
    
    @staticmethod
    def build_power2_conv_layers(max_filters, conv_filter_min, patch_size):
        filters = []
        dim = max_filters
        stride = 1
        while dim >= conv_filter_min:
            if stride < patch_size:
                filters.append(ConvParams(dim, 1, 1))
                filters.append(ConvParams(dim, 2, 2))
                stride = stride * 2
            else:
                filters.append(ConvParams(dim, 1, 1))
            dim = int(dim / 2)
        return filters
    
      
    @staticmethod
    def build_power2_1x1_conv_layers(conv_filter_max, conv_filter_min):
        enc_filters = []
        dec_filters = []
        dim = conv_filter_max
        while dim >= conv_filter_min:
            enc_filters.append(ConvParams(dim, 1, 1))
            dec_filters.insert(0, ConvParams(dim, 1, 1))
            dim = int(dim / 2)
            
        return enc_filters, dec_filters[1:]
    
    
    @staticmethod
    def build_power2_3x1_conv_layers(conv_filter_max, conv_filter_min, conv_filter_step, kernel, stride, indices_for_3x1):
        enc_filters = []
        dec_filters = []
        dim = conv_filter_max
        i = 0
        step = 0
        while dim >= conv_filter_min:
            if step % conv_filter_step == 0: 
                if i in indices_for_3x1:
                    enc_filters.append(ConvParams(dim, 1, 1))
                    if i + 1 in indices_for_3x1:
                        dec_filters.insert(0, ConvParams(dim, 1, 1))
                    else:
                        dec_filters.insert(0, ConvParams(dim, kernel, stride))
                else:
                    enc_filters.append(ConvParams(dim, kernel, stride))
                    # depends if we start with 2x1 or with 1x1
                    # start with 2x1
                    #dec_filters.insert(0, ConvParams(dim, 1, 1))      
                    # start with 1x1 
                    dec_filters.insert(0, ConvParams(dim, kernel, stride))     
               
                i = i + 1
            step = step + 1
            dim = int(dim / 2)
                
        return enc_filters, dec_filters[1:]
    
    
class Experiment():
    
    def __init__(self, experiment, scaler = None, suffix = ""):
        params = experiment.split("_")
        self.file = "_".join(params[0:2])
        self.patch_size = int(params[2])
        self.scaler = scaler
        self.suffix = suffix
        
    def build(self, n_features, path, experiment_with_suffix):
        return self
        

    def split_x_y(self, patches):
        self.x, self.y = patches.x_mean[patches.poi], patches.y_mean[patches.poi]
        
        return self
        
    def set_data(self):
        self.z = self.x

        return self

    def set_reduced_data(self, features):
        mz_idx = np.random.choice(self.x.shape[1], features, replace=False)
        self.z = self.x[..., mz_idx]
        
        return self
    
    
    def transform(self, z):
        if self.scaler is None:
            return z
        return self.scaler.transform(z)
        
    
    def get_mean_data(self, patches):
        mean_data = np.mean(patches, axis = (1,2))
        if self.scaler is not None:
            mean_data = mean_data.reshape(-1, 1)
            self.scaler.partial_fit(mean_data)
        return mean_data
   
    

class ConvAEExperiment(Experiment):
    def __init__(self, experiment, scaler = None, suffix = ""):
        super().__init__(experiment, suffix, scaler)
        params = experiment.split("_")
        self.activation = params[3]
        self.kernel = int(params[5])
        self.stride = int(params[6])
        self.conv_filter_max = int(params[7])
        self.conv_filter_min = int(params[8])
        self.conv_filter_step = int(params[9])
        self.samples_no = "_".join(params[10:14])
        self.mode = params[15]
        
        self.enc_conv_params, self.dec_conv_params = ConvParams.build_power2_3x1_conv_layers(self.conv_filter_max, self.conv_filter_min, self.conv_filter_step, self.kernel, self.stride, [0])
        
        
    def build(self, n_features, path, experiment_with_suffix):
                       
        conv_ae = ConvAE(n_features, self.patch_size, self.enc_conv_params, self.dec_conv_params, self.activation).build(False)
        conv_ae.encoder.load_weights(path + "weights/" + experiment_with_suffix + '_encoder_model_weights.h5') 
        conv_ae.decoder.load_weights(path + "weights/" + experiment_with_suffix + '_decoder_model_weights.h5')
       
        self.conv_ae = conv_ae
        return self
    
    def build_z(self, patches):
        z = self.conv_ae.encoder.predict(patches, verbose=0) 
        _ = gc.collect()
        mean_z = np.mean(z, axis = (1,2))
        return mean_z
        
        
    def split_x_y(self, patches):
        self.x, self.y = patches.x[patches.poi], patches.y_mean[patches.poi]
        return self
    
    def set_data(self):
        self.z = self.build_z(self.x)
        return self
        
        
    def get_mean_data(self, patches):
        mean_z = self.build_z(patches)
        self.scaler.partial_fit(mean_z)
        return mean_z
        