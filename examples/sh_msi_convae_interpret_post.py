import argparse
import os

import numpy as np
import pandas as pd

from datetime import datetime

from automsi import ae_preprocessing, ae_vae, ae_images, ae_recover, ae_utils


import h5py

PATCH_SIZE_IRRELEVANT = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad_path', type=str, help='Path to h5ad files')
    parser.add_argument('--ae_path', type=str, help='Path to experiment')
    parser.add_argument('--ms_ms_path', type=str, help='Path to results of tandem MS experiment, expecting tab-separated txt file.')
    parser.add_argument('--peaks_meta_path', type=str, help='Path containing raw msi information, expecting h5 file containing mean "_spectra" and "_mz" record per sample.')
    parser.add_argument('--peaks_path', type=str, help='Reference peaks used for pick peaking, expecting h5 file containing "peaks" record.')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived.')
    parser.add_argument('--scaler', type=str, help='Scaler to normalize spatial data.')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names to encode with the autoencoder, separated by ;')
    parser.add_argument('--latent_feature_id', type=int, help='Id of latent feature to recover information.')
    parser.add_argument('--backfill', type=str, default="ones", help='Noise values to fill features.')
    parser.add_argument('--cor_latent_space', type=float, default=0.95, help='Spearman correlation coefficient cutoff to define features as relevant.".')
    parser.add_argument('--cor_ion_images', type=float, default=0.80, help='Spearman correlation coefficient among different ion images to considered them as similar.')
    parser.add_argument('--conv_ae_prefix', type=str, default="cor_info_ones_feature_", help='Prefix name of h5 files containing correlation information, only used if mode == conv_ae.')
    parser.add_argument('--mass_min_error', type=float, default=0.0487, help='Minimum error between masses, usually the technical error between m/z values.')
    return parser.parse_args()


def match_with_reference(candidates, ref):
    matches = []
    for c in candidates: 
        if c in ref:
            matches.append(c)
            
    return matches


def main():     
    msi_masses = ae_recover.MSIMassCollection(peaks, mz, spectra, args.mass_min_error).gather_mz_ranges()
    matched_masses = ae_recover.TandemMS(evidence).reset_mass_for_modified_peptides("C", 0.0).find_all_candidate_peptides(msi_masses).summarize()

    cor, cor_features = ae_recover.summarize_features_correlating_with_latent_feature(result_path, args.conv_ae_prefix, n_features, 500, args.latent_feature_id, args.cor_latent_space, np.greater)
    print("Masses associated with latent feature", len(cor_features))
  
    peptide_candidates = ae_recover.PeptideCandidates(matched_masses, cor_features, args.cor_ion_images)
    peptide_candidates.find_correlations_between_ion_images(msi.train.x.images).match(all_msi_masses)
    peptide_candidates.filter_potential_mass_shifts(args.mass_min_error).print_results()
    
    hypoxia_match = match_with_reference(peptide_candidates.multiple_matches, ae_utils.HYPOXIA_GENES)
    print("Matches with hypoxia genes", hypoxia_match)
    
    today = datetime.today().strftime('%Y-%m-%d')
    suffix = str(args.cor_ion_images) + "_" + str(args.cor_latent_space)
    
    summary = pd.DataFrame(peptide_candidates.matches.items())
    with pd.ExcelWriter(result_path + today + "_" + exp.file + '_peptide_candidates_all_' + suffix + '.xlsx') as writer:  
        summary.to_excel(writer, index=False)
        
    summary = pd.DataFrame(peptide_candidates.multiple_matches.items())
    with pd.ExcelWriter(result_path + today + "_" + exp.file + '_peptide_candidates_multiple_' + suffix +  '.xlsx') as writer:  
        summary.to_excel(writer, index=False)
        
    summary = pd.DataFrame(peptide_candidates.mass_shifts.items())
    with pd.ExcelWriter(result_path + today + "_" + exp.file + '_peptide_candidates_mass_shifts_' + suffix + '.xlsx') as writer:  
        summary.to_excel(writer, index=False)

        
if __name__ == '__main__':
    args = parse_args()
    exp = ae_vae.ConvAEExperiment(args.experiment)
    result_path = os.path.join(args.ae_path, args.experiment, "")

    mz, spectra = ae_recover.load_raw_msi_information(args.peaks_meta_path, args.samples)
    peaks = ae_recover.load_ref_peaks(args.peaks_path)
    all_msi_masses = ae_utils.mz_values_to_masses(peaks)
    evidence = pd.read_csv(args.ms_ms_path, sep='\t')     

    ## msi data
    adata, _, n_features = ae_preprocessing.read_msi_from_adata(args.h5ad_path + exp.file + ae_utils.H5AD_SUFFIX + ".h5ad")
    samples = adata[adata.obs["batch"].isin(args.samples)]
    
    samples, _ = ae_preprocessing.normalize_train_test_using_scaler(samples, None, args.ae_path + "weights/" + args.scaler, pd.DataFrame(), None)
    
    msi = ae_images.MSIDataset(PATCH_SIZE_IRRELEVANT, n_features).build(samples, None, create_patches = False)
    
    main()