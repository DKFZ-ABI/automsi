import argparse

import glob
from automsi import ae_import


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msi_import', type=str, help='Path to imzml files containing peaks')
    parser.add_argument('--msi_export', type=str, help='MSI file name, h5ad file without file suffix')
    parser.add_argument('--log-transform', type=bool, default=False, help='Whether to log transform the msi data')
    parser.add_argument('--tol', type=str, help='Part of suffix to describe tolerance used during peak picking')
    
    parser.add_argument('--min_overlap', type=str, help='Part of suffix to describe number of overlapping samples used during peak picking ')
    parser.add_argument('--SNR', type=str, help='Part of suffix to describe signal to noise ratio used during peak picking')
    parser.add_argument('--prefix', type=str, help='Start of suffix to describe data')
    return parser.parse_args()


 
    
def main(args): 
    
    file_export = args.prefix + args.SNR + "o" + args.min_overlap + "t" + args.tol
    if args.log_transform:
        path_export = args.msi_export + file_export + "_log"
    else:
        path_export = args.msi_export + file_export 

    msi_prefix = [args.prefix + "_" + args.SNR + "_" + args.min_overlap + "_" + args.tol]
    imzML = glob.glob(args.msi_import + "*.imzML")
    imzML = [s for s in imzML if any(xs in s for xs in msi_prefix)]

    print(f"Detected files with prefix {msi_prefix }: {len(imzML)}")
    print(imzML.sort())

    all_data = ae_import.normalize(imzML, True, args.log_transform)
    all_data.write(path_export + "_nlabels.h5ad")
    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
