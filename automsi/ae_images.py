import numpy as np

import pandas as pd
import anndata as ad

from sklearn import preprocessing
import tensorflow as tf

import matplotlib.pyplot as plt

class SpatialImages():
    def __init__(self, n_features: int, padding_base: int, im_label: dict, obs_label: dict, obs_xy: list,
                 background: int):
        self.n_features = n_features
        self.padding_base = padding_base 
        self.im_label = im_label
        self.obs_label = obs_label
        self.obs_xy = obs_xy
        self.background = background
    
 
    def unfold_all_images(self, adata : ad.AnnData) -> pd.Series :
        batches = adata.obs.batch.value_counts()
        
        images = []
        names = []
        labels = []
        for name, value in batches.items():
            batch = adata[adata.obs.batch == name]
            im_range, xy_min = self.set_image_range(batch)
            im, l = self.unfold_image(batch, im_range, xy_min)
            images.append(im)
            labels.append(l)
            names.append(name)
        
        return pd.Series(images, index = names), pd.Series(labels, index = names)
    
    
    def set_image_range(self, batch : ad.AnnData):
        
        sample = batch.obs[self.obs_xy]
        xy_max = sample.max()
        xy_min = sample.min() # need to be passed, default 0
        im_range = xy_max - xy_min + 1

        padded_size = self.padding_base * np.ceil(im_range.max()/self.padding_base).astype(int)
        im_range = (padded_size, padded_size) # extract to other method and pass
        
        return im_range, xy_min
    
    def unfold_image(self, batch: ad.AnnData, im_range: tuple, xy_min: tuple):
        sample = batch.obs[self.obs_xy]
        data = batch.to_df()

        im = np.full((*im_range, self.n_features), self.background, dtype=float) 
        l = np.full((*im_range, len(self.im_label)), 0, dtype = float) 

        for i in range(sample.shape[0]):
            xy = sample.iloc[i] - xy_min
            
            if not np.all(np.isnan(data.iloc[i])):
                im[tuple(xy)[::-1]] = data.iloc[i]
            
            for j in self.im_label:
                l[tuple(xy)[::-1]][self.im_label[j]] = batch.obs[self.obs_label[j]].iloc[i]
        
        return im, l



class Patches():
    def __init__(self, patch_size : int, n_features : int):
        self.patch_size = patch_size
        self.no_patches = {}
        self.n_features = n_features
    
    def create_all(self, images : pd.Series):
        images_patches = []
        for name, im in images.items():
            p, no_p = self.create(im, self.patch_size)
            images_patches.append((name, p))
            self.no_patches[name] = no_p
            
        return pd.Series(dict(images_patches))
    
    
    def create_all_with_stride(self, images : pd.Series, stride: int):
        images_patches = []
        for name, im in images.items():
            p, no_p = self.create(im, stride)
            images_patches.append((name, p))
            
        return pd.Series(dict(images_patches))
    
    
    def create(self, image: np.ndarray, stride: int):
        #assert self.n_features == image.shape[2], "Expected different number of dimensions for n_features of image."
        patch_tensor = [1, self.patch_size, self.patch_size, 1] # without n_features    
        padded_size = image.shape[0]
        no_patches = int(padded_size * padded_size / (self.patch_size * self.patch_size)) 
        strides =  [1, stride, stride, 1]
        patches = tf.image.extract_patches(images = tf.expand_dims(image, 0),
                        sizes = patch_tensor,
                        strides = strides,
                        rates = [1, 1, 1, 1], padding='VALID')

        patches = tf.reshape(patches, (patches.shape[1]*patches.shape[2], self.patch_size, self.patch_size, image.shape[2]))
        return patches, patches.shape[0]
    
        
    def plot_full_image(self, image, name, mz_value, path_prefix = "", inverse = False):
        img = 1. - image[...,mz_value] if inverse else image[...,mz_value]
        plt.figure(figsize=(4, 4))
        
        plt.imshow(img, vmin=0, vmax=1)
        plt.title(name)
        plt.colorbar()
        
        if path_prefix != "": plt.savefig(path_prefix + "_" + str(mz_value) + ".png", dpi = 300, facecolor = "white")
        else: plt.show()
        
    def shape_with_no_patches_and_mz(self, patch, name, patch_size, n_features):
        patch = np.reshape(patch, (self.no_patches[name], patch_size, patch_size, n_features))
        n = int(np.sqrt(patch.shape[0]))
        return patch, n
    
       
    def show_patch(self, image, patch, mz_value = 1):
        plt.figure(figsize=(4, 4))
        plt.imshow(image[patch,...,mz_value], vmin=0, vmax=1)
        plt.colorbar()
        
        
    def show_patch_in_image(self, image, max_p = -1,  mz_value = 1, path_prefix = "", plot_show = True, inverse = False):
        if max_p == -1: max_p = image.shape[0] - 1
        n = int(np.sqrt(image.shape[0]))
        
        #plt.figure(figsize=(10, 10))
        for i in range(0, max_p + 1):
            ax = plt.subplot(n, n, i + 1)
            img = 1. - image[i,...,mz_value] if inverse else image[i,...,mz_value]
            plt.imshow(img, vmin=0, vmax=1)
            plt.axis("off")
        if path_prefix != "": plt.savefig(path_prefix + "_" + str(mz_value) + "_patched_" + str(max_p) + ".png", dpi = 300, facecolor = "white")
        if plot_show: plt.show()

        
    def show_patched_image(self, patches : np.ndarray, name : str, mz_value = 1, max_p = -1, path_prefix = ""):
        patch = patches[name]   
        (_, image) = self.get_image_from_patch(patch, name)
        self.plot_full_image(image, name, mz_value, path_prefix)
        
        self.show_patch_in_image(patch, mz_value, max_p, path_prefix)
        
  
    def get_image_from_patch(self, patch : pd.Series, name : str):
        patch, n = self.shape_with_no_patches_and_mz(patch, name, patch.shape[2], patch.shape[3]) 
        
        image = Patches.reshape_patch(patch, n)
        return (name, image)
    
        
    def get_labeled_images_from_patches(self, blueprint : pd.Series, patches : np.ndarray):
        index_split = np.cumsum([p.shape[0] for p in blueprint])
        # caution, assuming patches are symmetric
        value_split = np.split(patches, index_split[:-1])
        patches = pd.Series(value_split, index = blueprint.index)
        
        return patches, self.get_images_from_patches(patches)
    
    
    def get_images_from_patches(self, patches : pd.Series):
        rebuild_images = [self.get_image_from_patch(p, name) for name, p in patches.items()]
        return pd.Series(dict(rebuild_images))
    
    @staticmethod
    def reshape_patch(patch, n):
        rows = np.split(patch, n, axis=0) # n x (n, patch_size, patch_size, n_features)
        rows = [np.concatenate(np.moveaxis(x, 0, 0), axis = 1) for x in rows] # n x (patch_size, patch_size * n, n_features)
        image = np.concatenate(rows, axis = 0) # (patch_size * n, patch_size * n, n_features)
        return image
        
    
    @staticmethod
    def rebuild_image_directly(sample, feature_id):
        patch = np.reshape(sample[...,feature_id], (sample.shape[0], sample.shape[1], sample.shape[2], 1))
        n = int(np.sqrt(patch.shape[0]))
        
        return Patches.reshape_patch(patch, n)
    
    @staticmethod
    def rebuild_images_from_indices(patches, index_split, feature_id):
        # last index not needed for split
        series = np.split(patches, index_split[:-1])
        images = [Patches.rebuild_image_directly(sample, feature_id) for sample in series]
        return images

    
class SpatialDataset():

    def __init__(self, patch_size: int, n_features:int,
                 im_label: dict, obs_label: dict, obs_xy: list, background:int = 0):
        self._images = SpatialImages(n_features, patch_size, im_label, obs_label, obs_xy, background)
        self._patches = Patches(patch_size, n_features)
        self._yatches = Patches(patch_size, len(im_label))
        self._train = None
        self._test = None

    @property
    def train(self):
        if self._train is None:
            raise Exception('Dataset is not build yet.')
        return self._train

    @property
    def test(self):
        if self._test is None:
            raise Exception('Dataset is not build yet.')
        return self._test
    

    def build(self, train, test, create_patches = True):
        train_images, train_labels = self._images.unfold_all_images(train)
        self._train = SpatialData(self._patches, train_images, self._yatches, train_labels, create_patches)
            
        if test is not None:
            test_images, test_labels = self._images.unfold_all_images(test)
            self._test = SpatialData(self._patches, test_images, self._yatches, test_labels)
        return self
    
    def overlapping_patches(self, stride:int, on_test: bool = False):
        self._train = self._train.create_overlapping_patches(stride)
        if on_test:
            self._test = self._test.create_overlapping_patches(stride)
        return self

        
class MSIDataset(SpatialDataset):
    
    def __init__(self, patch_size: int, n_features:int,
                 im_label: dict = {'HE': 0, 'FI': 1}, obs_label: dict = {"FI" : "fi", "HE": "he"}, obs_xy: list = ["xLocation", "yLocation"], background:int = 0):
        super().__init__(patch_size, n_features, im_label, obs_label, obs_xy, background)
  
    
class SpatialDataRep():
    def __init__(self, images, patches, patches_overlap = None):
        self.images: SpatialImages = images
        self.patches = patches
        self._patches_overlap = patches_overlap
    
    @property
    def patches_overlap(self):
        if self._patches_overlap is None:
            raise Exception('Patches have not build yet.')
        return self._patches_overlap
    
    def flatten(self):
        patches = self.patches if self._patches_overlap is None else self.patches_overlap
        return np.concatenate(patches)
    
    def identify_non_zero_patches(self):
        patches = self.flatten()
        mean_value = np.mean(patches, axis = (1,2,3))
        del patches
        return np.where(mean_value != 0.)[0]
    

class SpatialData():

    def __init__(self, patches: Patches, images:list, yatches:Patches, labels:list, create_patches = True):
        self._patches = patches
        self._yatches = yatches
        if create_patches:
            self.x = SpatialDataRep(images, self._patches.create_all(images))
            self.y = SpatialDataRep(labels, self._yatches.create_all(labels))
        else: 
            self.x = SpatialDataRep(images, None)
            self.y = SpatialDataRep(labels, None)
        self.z = None
        self.dec = None
        
        
    def get_non_empty_patches(self):
        non_zero_p = self.x.identify_non_zero_patches()
        x, y = self.x.flatten(), self.y.flatten()
        return x[non_zero_p], y[non_zero_p]

    # avoid concatening huge x patch shapes
    def get_non_empty_idx_patches(self):
        idx = []
        x = self.x.patches if self.x._patches_overlap is None else self.x.patches_overlap
        y = self.y.flatten()
        i = 0
        for sample in x.items():
            for p in sample[1]: # loop through values
                p_mean = np.mean(p, axis = (0,1,2))
                if p_mean != 0.: 
                    idx.append(i)
                i = i + 1
        
        return x, y, idx

    def create_overlapping_patches(self, stride:int):
        if stride != 0:  # != patch_size
            # should be smaller than patch size
            self.x._patches_overlap = self._patches.create_all_with_stride(self.x.images, stride)
            self.y._patches_overlap = self._patches.create_all_with_stride(self.y.images, stride)
        else:
            self.x._patches_overlap = self.x.patches
            self.y._patches_overlap = self.y.patches
        return self
        
        
    def set_z_and_dec(self, args):
        z, dec_patches = args
        dec_patches, dec_images = self._patches.get_labeled_images_from_patches(self.x.patches, dec_patches)
        self.z = z
        self.dec = SpatialDataRep(dec_images, dec_patches)
        
    def clear(self):
        self.x = None
        self.y = None

