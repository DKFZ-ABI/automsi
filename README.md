# automsi

automsi provides a convolutional autoencoder implementations for MSI data. 


## Installation

```bash
pip install automsi
```

## Usage

The package is setup in a modular way, with examples provided for every step. The typically workflow would look like this:

- Setup an [anndata](https://anndata.readthedocs.io/en/latest/) structure from imzML files of multiple samples. See [sh\_msi\_imzml\_to\_ann.py](examples/sh_msi_imzml_to_ann.py)
- Add some continuous annotations (like from fluorescence images or H\&E stains) from text files. See [sh_msi_ann_annotate.py](examples/sh_msi_ann_annotate.py)
- Train the autoencoder, either in an unsupervised or semi-supervised mode. See [sh_msi_convae](examples/sh_msi_convae.py) on the anndata.
- Visualize the latent space of your autoencoder. See [sh_msi_convae_post](examples/sh_msi_convae_post.py)
- Run a random forest regression model to derive which  features show a high feature importance for your provided annotations (e.g., hypoxia). Features may be either original features (random forest only approach) or latent features from the ConvAE approach.
See [sh_convae_regression](examples/sh_convae_regression.py)
- Recover which original features (m/z values) contributed to a latent feature of interest. See [sh_convae_interpret](examples/sh_convae_interpret.py)
- Complement the associated original features of MSI with tandem MS experiments. See [sh_msi_convae_interpret_post](examples/sh_msi_convae_interpret_post.py)
- Derive the [structural similarity index measure](https://en.wikipedia.org/wiki/Structural_similarity/) for a set of features (e.g., associated with hypoxia) and a known reference. Set of features may have been derived from a random forest only approach or a ConvAE approach. See [sh_metrics_msi_convae](examples/sh_metrics_msi_convae.py)
- Visualize arbitrary aspects like ion images, reconstructed images, annotation images for a set of samples. See [plt_msi_convae](examples/plt_msi_convae.py)


## Autoencoder

automsi enables to create a convolutional autoencoder with a variable number of hidden layers, based on the parameters passed.

The parameter *conv_filter_step* defines the steps in power two to take between *conv_filter_min* and *conv_filter_max*, e.g. from 8 to 1024 are 7 steps: 1024 > 512 > 256 > 128 > 64 > 32 > 16 > 8.

```python
from automsi import ae_vae

conv_filter_max = 1024
conv_filter_min = 8
conv_filter_step = 7
kernel = 2
stride = 1
indices_with_zero_stride = [0]
enc_conv_params, dec_conv_params = ae_vae.ConvParams.build_power2_3x1_conv_layers(conv_filter_max, conv_filter_min, conv_filter_step, kernel, stride, indices_with_zero_stride
```


In this case, two hidden layers are used for encoding, one with 1024 neurons and one with 8 neurons. Setting *conv_filter_step = 3* would lead to 3 hidden layers (1024, 128, 3).

The last parameters defines the layers with stride being set to 1. For example, [0] would indicate that the patch size between the input and the first hidden layers remains unchanged. The kernel size is defined globally. 


The actual convolutional autoencoder is then created by using the command:


```python
n_features = 18735
patch_size = 3
activation = "sigmoid"
conv_ae = ae_vae.ConvAE(n_features, patch_size, enc_conv_params, dec_conv_params, activation).build() 
```




## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements
automsi was developed under the umbrella of the the Division of Applied Bioinformatics and Division of Radiooncology / Radiobiology at the [Germany Cancer Research Center (DKFZ)](https://www.dkfz.de).

The present contribution is supported by the Helmholtz Association under the joint research school "[HIDSS4Health – Helmholtz Information and Data Science School for Health](https://www.hidss4health.de/)".


If you use this software, we appreciate if you cite the following paper (currently available as preprint via Research Square):

Verena Bitto, Pia Hönscheid, María José Besso et al. Easing accessibility to mass spectrometry imaging using convolutional autoencoders for deriving hypoxia-associated peptide candidates from tumors, 22 January 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3755587/v1]
