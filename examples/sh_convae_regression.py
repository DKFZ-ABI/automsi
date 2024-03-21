import argparse
import os

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
import pandas as pd
from automsi import ae_preprocessing, ae_vae, ae_images, ae_utils, ae_rf
import random
from sklearn.model_selection import cross_validate, ShuffleSplit


N_TREES = 1000
MTRY = "sqrt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad_path', type=str, help='Path to h5ad files')
    parser.add_argument('--ae_export', type=str, help='Path to experiment')
    parser.add_argument('--experiment', type=str, help='Name to experiments from which autoencoder parameters can be derived.')
    parser.add_argument('--scaler', type=str, help='Scaler to normalize MSI data.')
    parser.add_argument('--samples', type=ae_utils.split_at_semicolon, help='Sample names to train the RF regressor, separated by ;')
    parser.add_argument('--label', type=str, default="FI", help='Label used for training RF regressor.')
    parser.add_argument('--overlapping_patches', type=int, default=1, help='Stride information for overlapping patches.')
    parser.add_argument('--cutoff', type=float, help='Mean cutoff value to define a single patch as being hypoxic (or any other label), value between [0, 1].')
    parser.add_argument('--other_fraction', type=float, help='Defines the fraction of non-hypoxic patches to be sampled, value between [0, 1].')
    parser.add_argument('--mode', type=str, default="conv_ae", help='Whether to encode patches before training RF regressor or not (conv_ae or original).')
    parser.add_argument('--features', type=int, default=18735, help='Number of features to use, for original mode only.')
    parser.add_argument('--scoring', type=str, default="r2", help='Scoring metric for RF regressor.')
    parser.add_argument('--suffix_from', type=str,  default="_001", help='Can be used to evaluate multiple experiments at once, expected to be of kind "_{0-9}3".')
    parser.add_argument('--suffix_to', type=str,  default="_001", help='Can be used to evaluate multiple experiments at once, expected to be of kind "_{0-9}3".')
        
    return parser.parse_args()
   


def extract_feature_importance(classifier, feature_names, z, y, suffix, path_prefix, args):
    top = []
    fi_path = path_prefix  + "_rf_" if args.mode == "original" else  path_prefix  + "/rf_"
    writer = pd.ExcelWriter(fi_path + str(args.overlapping_patches) + suffix + '.xlsx')
    for idx, estimator in enumerate(classifier['estimator']):
        feature_importances_ = estimator.feature_importances_
            
        feature_importances = pd.DataFrame(feature_importances_,
                                           index = feature_names,
                                           columns=['importance']).sort_values('importance', ascending=False)
    
        
        feature_importances.to_excel(writer, index=True,  sheet_name=str(idx))
        top.append(feature_importances[0:1])
    writer.close()
    
    top = pd.concat(top)
    top_index = top.index.value_counts().nlargest(1)
    top_importance = top["importance"].iloc[np.where(top.index == top_index.index[0])[0]]
    
    print("Highest feature importance: ")
    print(top_index)
    
    return top_index, np.mean(top_importance)



def run_regression(exp, suffix, experiment_with_suffix, args):
    print("Print some y values")
    print(exp.y[0:10])
    path_prefix = os.path.join(args.ae_export, experiment_with_suffix)
    
    random_state = random.randint(1, 1000)
    forest = ae_rf.RandomForestRegressionBuilder(n_estimators = N_TREES, max_features = MTRY, random_state = random_state) 
    
    shuffle = ShuffleSplit(n_splits=10, random_state=42, test_size = 0.33)
    classifier = cross_validate(forest.forest, exp.z, exp.y, scoring=args.scoring, cv=shuffle, return_estimator =True)
    
    all_trees = []
    for fold_estimator in classifier['estimator']:
        individual_estimators = fold_estimator.estimators_
        all_trees.extend(individual_estimators)
    
    # Get the depth
    mean_depth = sum(tree.get_depth() for tree in all_trees) / len(all_trees)
    min_depth = min(tree.get_depth() for tree in all_trees)
    max_depth = max(tree.get_depth() for tree in all_trees)
    print(f"The mean depth of all trees in the Random Forest is: {mean_depth}")
    print(f"The min depth of all trees in the Random Forest is: {min_depth}")
    print(f"The max depth of all trees in the Random Forest is: {max_depth}")
    
    print(f"RF test score: {np.mean(classifier['test_score']) :.3f}")
       
    feature_names = np.array(range(exp.z.shape[1]))
    top_index, top_imp = extract_feature_importance(classifier, feature_names, exp.z, exp.y, suffix, path_prefix, args)

    score = ae_rf.RandomForestSummary(suffix, 
                                        exp.z.shape[0], np.mean(classifier['test_score']),
                                        top_index.index[0], top_imp, int(top_index.values))
    return score


def run_experiments(exp, patch_adapter, suffix, args):
    experiment_with_suffix = args.experiment + "_" + suffix
    
    print(experiment_with_suffix)
    exp.build(n_features, args.ae_export, experiment_with_suffix).split_x_y(patch_adapter)

    if args.mode == "conv_ae" or n_features == args.features:
        exp.set_data()
    elif args.mode == "original":
        exp.set_reduced_data(args.features)
        
    score = run_regression(exp, suffix, experiment_with_suffix, args)
    return score

def set_patch_adapter(args):
    spatial = ae_images.MSIDataset(exp.patch_size, n_features, im_label = {'FI': 0}, obs_label = {"FI" : "fi"})
    spatial.build(spt_samples, None, create_patches = False).overlapping_patches(args.overlapping_patches, on_test = False)
    x, y, idx = spatial.train.get_non_empty_idx_patches()
    print("Shape of patches", x.shape, y.shape)
    
    patch_adapter = ae_rf.PatchAdapter(x, y, idx, spatial._images.im_label)
    patch_adapter.undersample_non_label_patches(args.label, args.cutoff, args.other_fraction).reduce_corresponding_x_patches_to_mean()
    
    return patch_adapter


def main(exp, args):
    patch_adapter = set_patch_adapter(args)
    scores = []   

    for i in range(int(args.suffix_from[1:]), int(args.suffix_to[1:]) + 1):
        suffix = "_" + str(i).zfill(3)
        score = run_experiments(exp, patch_adapter, suffix, args)
        scores.append(score)
                   
    global_naming = [args.experiment] + list(vars(args).values())[5:14]
    global_naming = '_'.join(str(n) for n in global_naming) 
    
    df = pd.DataFrame([vars(f) for f in scores])
    with pd.ExcelWriter(args.ae_export + global_naming + "_scores.xlsx") as writer:
        df.to_excel(writer, index=False)



if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == "conv_ae": 
        exp = ae_vae.ConvAEExperiment(args.experiment)
        #exp = ae_vae.ConvVAEExperiment(args.experiment)
    elif args.mode == "original":
        exp = ae_vae.Experiment(args.experiment)
    
    print("Loading patches from: " + exp.file)
    adata, _, n_features = ae_preprocessing.read_msi_from_adata(args.h5ad_path + exp.file + ae_utils.H5AD_SUFFIX + ".h5ad")
    spt_samples = adata[adata.obs["batch"].isin(args.samples)]
    scaler = args.ae_export + "weights/" + args.scaler
    spt_samples, _ = ae_preprocessing.normalize_train_test_using_scaler(spt_samples, None, scaler, spt_samples.obs, None)
    
    main(exp, args)
    
