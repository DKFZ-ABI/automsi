import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tensorflow.keras.callbacks import History


def generate_ion_image(adata: ad.AnnData):
    batches = adata.obs.batch.value_counts()
    
    for name, value in batches.iteritems():
        batch = adata[adata.obs.batch == name]
        sample = batch.obs[["xLocation", "yLocation"]]
        data = batch.to_df()
        
        xy_max = sample.max()
        xy_min = sample.min()
        
        im_range = xy_max - xy_min + 1
        im = np.full(tuple(im_range)[::-1], np.nan)
        
        for i in range(sample.shape[0]):
            xy = sample.iloc[i] - xy_min
            im[tuple(xy)[::-1]] = data.iloc[i]
        yield im, data, name
       

def plot_history(h : History, metric : str, ylim : list, 
                 plot_val : bool = True, path_prefix : str = ""):
    plt.plot(h.history[metric])
    plt.ylabel(metric); plt.xlabel('epoch')
    plt.ylim(ylim)
    
    if plot_val:
        val = 'val_' + metric
        plt.plot(h.history[val])
        plt.legend(['train', 'val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')    
    
    if path_prefix != "": plt.savefig(path_prefix + "_history_" + metric + "_.png", dpi = 300, facecolor = "white")
    else: plt.show()
        
   
    plt.clf()
    plt.close() 
    

    
    
# expecting latent_z to be of shape (latent_dim, patch_size, patch_size, n_features)
def conv_generate_latent_plots_from_patches(latent_z : np.ndarray, 
                         max_rows_per_page : int, max_samples_per_cols : int,
                         file_name : str = ""):
    
    if file_name != "": pp = PdfPages(file_name)
    fig_size = (20,20)

    rows_per_latent = np.ceil(len(latent_z) / max_samples_per_cols).astype(int)
    max_rows = latent_z[0].shape[2] * rows_per_latent # expecting to need several rows for all samples
                            
    fig, axes = plt.subplots(nrows = max_rows_per_page, ncols = max_samples_per_cols, figsize = fig_size)
    page = 1
    print(max_rows)
    print(rows_per_latent)
    for row in range(0, max_rows, rows_per_latent):
        if(row % rows_per_latent == 0):  
            mz_i = (int) (row/rows_per_latent)
            
        if np.ceil((row+1)/max_rows_per_page).astype(int) > page:
            fig.tight_layout()
            if file_name != "": pp.savefig(fig)
            else: plt.show()
            plt.cla()
            page = page + 1 

        col = 0
        mz_row = 0
        for name, sample_latent_z in latent_z.items():
            axes[(row + mz_row) % max_rows_per_page, col % max_samples_per_cols].clear() 
            axes[(row + mz_row) % max_rows_per_page, col % max_samples_per_cols].imshow(sample_latent_z[...,mz_i])  # , cmap='hot'
            axes[(row + mz_row) % max_rows_per_page, col % max_samples_per_cols].set_title(name + ", latent_mz=" + str(mz_i))
            col = (col + 1) 
            if(col % max_samples_per_cols == 0):
                mz_row = mz_row + 1
        
    fig.tight_layout()
    if file_name != "":
        pp.savefig(fig)
        pp.close()
    else:
        plt.show()
        
    plt.close()
    



# classifier

def plot_patches_of_interest(y, z, latent_dim, patches_of_interest, path_prefix = ""):
    fig, axs = plt.subplots(ncols = 1, nrows = latent_dim, figsize=(15, 20))

    for latent, ax in enumerate(axs):
        color = [int(item*100) for item in y[patches_of_interest]]
        ax.scatter(z[patches_of_interest, latent], np.repeat(0, len(color)), marker='|', c=color, cmap = "coolwarm")

        ax.set_yticklabels([])
        ax.set(ylabel=latent)
    if path_prefix != "": plt.savefig(path_prefix + "_" + "patches_of_interest.png", dpi = 300, facecolor = "white")
    else: plt.show()

    plt.clf()
    plt.close()