import argparse

import glob
from automsi import ae_preprocessing, ae_import


def split_at_semicolon(value):
    return value.split(';')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msi_import', type=str, help='Path to msi_files')
    parser.add_argument('--msi_files', default="cal336o1t0_15", type=str, help='MSI file name, h5ad file without file suffix')
    parser.add_argument('--msi_suffix', type=str, default="_labels", help='Suffix to specific msi_files that contain annotations, e.g., _labels')
    parser.add_argument('--msi_images', type=str, help='Path to images, in which subdirectories for corresponding annotations are expected')
    
    parser.add_argument('--invalid_he_annotations', type=split_at_semicolon, help='Samples for which no HE annotations are available, separated by ;')
    parser.add_argument('--invalid_fi_annotations', type=split_at_semicolon, help='Samples for which no FI annotations are available, separated by ;')
    return parser.parse_args()


def main(args): 
    adata, mz_values, n_features = ae_preprocessing.read_msi_from_adata(args.msi_import + args.msi_files + args.msi_suffix + ".h5ad")     
    annotated_data = ae_import.annotate(adata, args.msi_images + "annotations_he/", args.msi_images + "annotations_fi/", args.invalid_he_annotations, args.invalid_fi_annotations, " ")
    annotated_data.write(args.msi_import + args.msi_files + "_labels.h5ad")

    

if __name__ == '__main__':
    args = parse_args()
    main(args)
