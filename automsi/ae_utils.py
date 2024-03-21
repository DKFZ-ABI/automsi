import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Suffix of msi files to denote data includes annotations
H5AD_SUFFIX = "_labels"

# Prefix of imzML files, used for import
SAMPLES = "Sample[0-9]{1}"

def split_at_semicolon(value):
    return value.split(';')

def split_int_at_semicolon(value):
    return [int(x) for x in split_at_semicolon(value)]

def mz_values_to_masses(peaks):
    return peaks * 1 - 1


HYPOXIA_GENES = ["KDM3A", "CA9", "ANKRD37", "NDRG1", "INSIG2", "P4HA1", "BNIP3", "MAFF", 
    "PPFIA4", "DDIT4", "KCTD11", "BNIP3L", "EGLN3", "VEGFA", "ERBB4", "PDK1", 
    "SYNGR3", "TAF7L", "ANGPTL4", "ENO2", "ADM", "GBE1", "TPI1", "PFKFB3", 
    "HK2", "CD44v6", "PHD2", "HPV16", "CXCL12", "SLC2A1", "ZNF654", "HILPDA", 
    "CXCR4", "FGFR1", "EPOR", "SLC5A1", "LDHA", "ERCC4", 
    "LILRB1", "ERO1L", "WSB1", "LOXL2", "MXI1", "BCL2L1", "PGK1", "FLT1", 
    "FGFR2", "KDR", "SERPINB2", "GLRX", "P4HA2", "MAPK3", "PDPK1", "KCNA5", 
    "KIF2A", "HOOK2", "EGLN1", "ERO1A"]

GENES_OF_INTEREST =  ["LDHA", "PGK1", "KRT5", "EIF3A", "P4HB", "ENO1"]


# used for variational autoencoder approach
def get_patches_total(index_split):
    patches_total = []
    for i in range(len(index_split) - 1, 0, -1):
        patches_total.append(index_split[i] - index_split[i - 1])

    patches_total.append(index_split[0])
    patches_total = patches_total[::-1]
    return patches_total

    
def rebuild_flat_images(patches, feature_id, index_split, patches_split):
    patches_total = get_patches_total(index_split)
    prev_index = 0
    
    images = []
    for no_patches, index in zip(patches_split, patches_total):
        im = np.reshape(patches[prev_index:prev_index + index,feature_id], (no_patches, no_patches, 1))
        images.append(im)
        prev_index = prev_index + index
    
    return images


# expecting latent_z to be of shape (latent_dim, n_features)
def conv_generate_latent_plots(latent_z : np.ndarray, 
                         sample_size : list, patches_total : list,
                         max_rows_per_page : int, max_samples_per_cols : int,
                         file_name : str = ""):
    
    if file_name != "": pp = PdfPages(file_name)
    fig_size = (20,20)

    rows_per_latent = np.ceil(len(sample_size) / max_samples_per_cols).astype(int)
    max_rows = latent_z.shape[1] * rows_per_latent # expecting to need several rows for all samples
                            
    fig, axes = plt.subplots(nrows = max_rows_per_page, ncols = max_samples_per_cols, figsize = fig_size)
    page = 1
    for row in range(0, max_rows, rows_per_latent):
        if(row % rows_per_latent == 0):  
            mz_i = (int) (row/rows_per_latent)
            
        if np.ceil((row+1)/max_rows_per_page).astype(int) > page:
            fig.tight_layout()
            if file_name != "": pp.savefig(fig)
            else: plt.show()
            plt.cla()
            page = page + 1 

        prev_index = 0
        col = 0
        mz_row = 0
        for sample, index in zip(sample_size, patches_total):
            name, size = sample
            latent_z_image = np.reshape(latent_z[prev_index:prev_index+index], (size, size, latent_z.shape[1]))
            axes[(row + mz_row) % max_rows_per_page, col % max_samples_per_cols].clear() 
            axes[(row + mz_row) % max_rows_per_page, col % max_samples_per_cols].imshow(latent_z_image[:,:,mz_i])  # , cmap='hot'
            axes[(row + mz_row) % max_rows_per_page, col % max_samples_per_cols].set_title(name + ", latent_mz=" + str(mz_i))
            prev_index = prev_index + index
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
    
def plot_latent_plots(latent_z, images, patches, patch_size, path_prefix: str = ""):
    max_rows_per_page = 6  # <= latent_dim
    max_samples_per_cols = 5 # <= n_samples
    
    sample_size = [(name, int(d.shape[0] / patch_size)) for name, d in zip(images.index, images)]
    patches_total = [d.shape[0] for d in patches]
    if path_prefix != "": path_prefix = path_prefix + ".pdf"
    conv_generate_latent_plots(latent_z, 
                      sample_size, patches_total,
                      max_rows_per_page, max_samples_per_cols,
                      path_prefix)