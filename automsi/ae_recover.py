import pandas as pd
import numpy as np
import h5py

import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.stats
#import statsmodels.stats.api as sms
from sewar.full_ref import ssim

from automsi.ae_images import SpatialImages

class MetricCollector:
    
    def __init__(self, ref_id):
        self.ref_id = ref_id
        
        
    def derive_ssim(self, images: SpatialImages, feature_ids: list):
        ssims_sample = []
        for j in range(0, len(images)):
            ref_image = images[j][..., self.ref_id] 
            ref_image = (ref_image * 255).astype(np.uint8)

            ssims = np.zeros(len(feature_ids))
            for i, idx in enumerate(feature_ids):
                ion_image = images[j][...,idx]
                ion_image = (ion_image * 255).astype(np.uint8)

                result = ssim(ion_image, ref_image)
                ssims[i] = result[0] 

            ssims_sample.append(ssims)
            print(images.index[j])

        return ssims_sample
        

class FeatureCollector:
    
    def __init__(self, cutoff, cross_vals = range(0,10)):
        self.cutoff = cutoff
        self.cross_vals = cross_vals
        
    def get_by_score(self, path, col_names, score):
        # collect only top-ranked features
        if self.cutoff == 1: 
            feature_set = []
        else:
            feature_set = set()
            
        for i in self.cross_vals:
            sheet = pd.read_excel(path, str(i), header=0, names=col_names) 
            top = sheet.iloc[0]
            idx_cut = np.argmin(sheet[score] >= top[score] / self.cutoff)
            feature_ids = sheet.mz_index.loc[0:idx_cut-1]
            if self.cutoff == 1:
                feature_set.append(feature_ids[0])   
            else:
                feature_set.update(feature_ids)   
                
        return np.array(list(feature_set))
        
        

def load_raw_msi_information(peaks_meta_path, samples):
    mz = []
    spectra = []

    with h5py.File(peaks_meta_path, "r") as f:
        for name in samples:
            spectra.append(np.array(f[name + "_spectra"][:]))
            mz.append(np.array(f[name + "_mz"][:]))
            
    return mz, spectra
      

def load_ref_peaks(peaks_path):    
    with h5py.File(peaks_path, "r") as f:
        peaks = f["peaks"][()]
        
    return peaks



def summarize_features_correlating_with_latent_feature(ae_path, prefix, n_vars, step, latent_id, threshold, fu):

    all_cor = []
    cor = np.zeros(n_vars)

    for idx_from in range(0, n_vars, step): 
        idx_to = idx_from + step
        idx_to = min(idx_to, n_vars)
        suffix = "_".join([str(latent_id), str(idx_from), str(idx_to)])
        try:
            with h5py.File(ae_path + prefix + suffix + ".h5", "r") as f:
                cor[idx_from:idx_to] = f["cor"][()]
        except OSError as e:
            print("Skipped file: " + ae_path + prefix + suffix)

    # need to reverse, such that positive correlations are ranked top
    cor_index = [index for index, value in sorted(enumerate(cor), reverse=True, key=lambda x: x[1])]
    cor_index_cut = np.asarray(cor_index)[np.where(fu(cor[cor_index], threshold))[0]]

    return cor, cor_index_cut
    
    



class MSIMass:
    
    def __init__(self, mz_value, mz_range_min, mz_range_max):
        self.mz_value = mz_value
        self.mz_range_min = mz_range_min
        self.mz_range_max = mz_range_max
        self.mass = mz_value * 1 - 1
        self.mass_range_min = mz_range_min * 1 -1
        self.mass_range_max = mz_range_max * 1 -1  





class MSIMassCollection():
    MINIMUM_PEAK_HEIGHT = 2
    MAXIMUM_MZ_ID = 53400
    
    def __init__(self, peaks: list, mz: list, spectra: list, min_error, peak_indices = None, initial_range = 5):
        self.peaks = peaks
        self.mz = mz
        self.spectra = spectra
        self.peak_indices = peak_indices
        self.min_error = min_error
        self.initial_range = initial_range
        if peak_indices == None:
            self.peak_indices = range(0, len(peaks))
  
            

    def __lin_interp(self, x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

                                      
    def __half_max_x(self, x, y):
        half = (y[self.initial_range]) /2

        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        # len(zero_crossings) == len(x) -2
        zero_crossings_i = np.where(zero_crossings)[0]

        if len(zero_crossings_i) > 2:
            # self.initial_range is of by one because of (zero_crossings) == len(x) -2
            first_id = np.abs(zero_crossings_i - (self.initial_range-1)).argmin()
            zero_crossings_i = zero_crossings_i[first_id:]

        if len(zero_crossings_i) < 2:
            return [], half
        return [self.__lin_interp(x, y, zero_crossings_i[0], half),
                self.__lin_interp(x, y, zero_crossings_i[0+1], half)], half

    
    def __set_mz_range(self, hmx, range_min, range_max):
        if range_min == 0 or range_max == 0:
            range_min = hmx[0]
            range_max = hmx[1]

        range_min = min(hmx[0], range_min)
        range_max = max(hmx[1], range_max)
        return range_min, range_max
                     
    


    def __check_if_within_mz_range(self, range_min, range_max, mz_value, log_borders):
        mz_range_min = mz_value - self.min_error
        mz_range_max = mz_value + self.min_error
        adjusted = False

        if range_min > mz_range_min:
            mz_range_min = range_min
            adjusted = True
            if log_borders: 
                print("Mz value " + str(mz_value) + " is not within defined error borders")
                print("Expected min borders: " + str(range_min) + " actual min boarders: " + str(mz_value - self.min_error) )

        if range_max < mz_range_max:
            mz_range_max = range_max
            adjusted = True
            if log_borders: 
                print("Mz value " + str(mz_value) + " is not within defined error borders")
                print("Expected max borders: " + str(range_max) + " actual max boarders: " + str(mz_value + self.min_error))

        return mz_range_min, mz_range_max, adjusted

                                      
                                      
    def __calculate_mz_range_from_multiple(self, mz_range, plot_fwhm, extensive_logging = False):
        range_min = 0
        range_max = 0

        palette = cm.Greys(np.linspace(0, 1, 6))

        for i, spec in enumerate(self.spectra):
            y = self.spectra[i][mz_range[i]]
            x = self.mz[i][mz_range[i]]

            if len(x):
                hmx, half = self.__half_max_x(x, y)
                if hmx:
                    fwhm = hmx[1] - hmx[0]
                    range_min, range_max = self.__set_mz_range(hmx, range_min, range_max)
                                      
                    if plot_fwhm:
                        plt.plot(x,y, color = palette[i])
                        plt.plot(hmx, [half, half], color = palette[i], linestyle='dashed')
            elif extensive_logging: 
                print("Skipped sample, because peak was either entirely missing or missed zero_crossing.")

        return range_min, range_max

    # for simplicity we take the mz values of the first sample as reference (self.mz[0])
    def __find_closest_mz(self, mz_value):
        mz_id = (np.abs(self.mz[0] - mz_value)).argmin()
        return mz_id


    def __set_up_mz_ranges(self, mz_id, extensive_logging = False):
        mz_ranges = []
        for spec in self.spectra:
            mz_range =  np.asarray(range(mz_id-self.initial_range, min(mz_id+self.initial_range, MSIMassCollection.MAXIMUM_MZ_ID-1)))
            loc_max = (np.diff(np.sign(np.diff(spec[mz_range]))) < 0).nonzero()[0] + 1    
            # there might be more than one maximum within mz_range, we chose the one which is closest to our mz id
            if len(loc_max):
                max_id = loc_max[np.abs(loc_max - mz_id).argmin()]
                if spec[mz_range[max_id]] > MSIMassCollection.MINIMUM_PEAK_HEIGHT: 
                    mz_range = np.asarray(range(mz_range[max_id]-self.initial_range, min(mz_range[max_id]+self.initial_range+1, MSIMassCollection.MAXIMUM_MZ_ID-1)))
                    mz_ranges.append(mz_range)
                    if extensive_logging: print(spec[mz_range])
                else:
                    mz_ranges.append([])
                    if extensive_logging: print("Skipped sample, because it failed to reach minimum peak height.")
            else:  
                mz_ranges.append([])
                if extensive_logging: print("Skipped sample, because it had no local maximum.")

        return mz_ranges                                      
                                      
    def gather_mz_ranges(self, plot_fmwh: bool = False, plot_errors: bool = False, log_borders: bool = False, extensive_logging: bool = False):
        msi_masses = []
        for peak_id in self.peak_indices: 
            mz_id = self.__find_closest_mz(self.peaks[peak_id])
            mz_ranges = self.__set_up_mz_ranges(mz_id, extensive_logging = extensive_logging)

            if len(mz_ranges): 
                range_min, range_max = self.__calculate_mz_range_from_multiple(mz_ranges, plot_fmwh, extensive_logging)
                if range_min <= self.peaks[peak_id] <= range_max:
                    mz_range_min, mz_range_max, adjusted = self.__check_if_within_mz_range(range_min, range_max, self.peaks[peak_id], log_borders)
                    msi_masses.append(MSIMass(self.peaks[peak_id], mz_range_min, mz_range_max))
                elif extensive_logging: print("Skipping mass, because peak was not within necessary space. Peak: " + str(self.peaks[peak_id]) + ", mz_id: " + str(self.mz[0][mz_id]))

                if plot_fmwh:
                    plt.axhline(y = 0, color = 'gray')    
                    plt.axvline(x = self.peaks[peak_id], ymin=0.04, color = "#1874CD")
                    if plot_errors:
                        plt.plot([range_min, range_max], [0, 0], color = "#FF3030")
                        plt.plot([self.peaks[peak_id] - self.min_error, self.peaks[peak_id] + self.min_error], [2, 2], color = "#00BFFF")
                        if range_min <= self.peaks[peak_id] <= range_max and adjusted:
                                plt.axvline(x = mz_range_min, ymin=0.04, ymax=0.20, color = "#FF3030", linestyle='dotted', linewidth = 2)
                                plt.axvline(x = mz_range_max, ymin=0.04, ymax=0.20, color = "#FF3030", linestyle='dotted', linewidth = 2)
            elif extensive_logging: print("Skipping masse, because mass range for peak could not be derived.")

        return msi_masses
                                      
                                      

class TandemMS():
                                      
    def __init__(self, collection):
        self.collection = collection
        self.candidates = None
                                    

    def reset_mass_for_modified_peptides(self, modification, mass_to_substract):
        for idx, row in self.collection.iterrows():
            modified_peptides = re.finditer(modification, row.Sequence)
            indices = [p.start() for p in modified_peptides]
            no_modifications = len(indices)
            if(no_modifications > 0):
                self.collection.loc[idx, "Mass"] = row.Mass - mass_to_substract * no_modifications
                                      
        return self
                                      
                                      
    def __isNaN(self, key):
        return key != key


    def __update_candidate_entry(self, candidates, match, idx, msi_mass, total, ion_charge):
        row = match.iloc[idx]
        key = row["Gene names"]
        if self.__isNaN(key): 
            key = row["Sequence"]
        if key not in candidates: candidates[key] = list()
        #else : print("key " + str(key) + " already exists in list")

        candidates[key].append(
            (msi_mass.mass,   
             1.0 / total, 
             row["Sequence"],
             row["Mass"],
             row["Protein names"],
             row["Experiment"],
             ion_charge)
        )


    def __diff_smaller_than_next_smaller_msi_mass(self, msi_masses, i, diff, tandem_mass):
        if i > 0:
            diff_msi = tandem_mass - msi_masses[i-1].mass
        else: 
            diff_msi = 1 # there is no smaller mass to compare

        best_match =  True if(abs(diff) < abs(diff_msi)) else False
        return best_match


    def __diff_smaller_than_next_bigger_msi_mass(self, msi_masses, i, diff, tandem_mass):
        if i < len(msi_masses)-1:
            diff_msi = tandem_mass - msi_masses[i+1].mass
        else: 
            diff_msi = 1 # there is no bigger mass to compare

        best_match =  True if(abs(diff) < abs(diff_msi)) else False
        return best_match
                                      

    def find_all_candidate_peptides(self, msi_masses):
        candidates = {}

        for i, msi_mass in enumerate(msi_masses):
            match = self.collection.loc[self.collection.Mass.between(msi_mass.mass_range_min, msi_mass.mass_range_max)]
            relevant_match = match[["Sequence", "Protein names", "Gene names", "Mass", "Experiment"]].drop_duplicates() 

            if(len(relevant_match) > 0):
                for idx in range(len(relevant_match)):
                    diff = msi_mass.mass - relevant_match.iloc[idx].Mass
                    if np.logical_and(self.__diff_smaller_than_next_smaller_msi_mass(msi_masses, i, diff, relevant_match.iloc[idx].Mass), self.__diff_smaller_than_next_bigger_msi_mass(msi_masses, i, diff, relevant_match.iloc[idx].Mass) ): 
                        self.__update_candidate_entry(candidates, relevant_match, idx, msi_mass, len(relevant_match), "") 
        self.candidates = candidates
        return self


    def summarize(self):
        all_genes = []
        for item in self.candidates.items():
            gene, details = item 
            df = pd.DataFrame(details, columns=["MSI mass", "1/no of matches", "Sequence", "MS/MS Mass", "Protein names", "Experiment", "Expected MSI Single Ion Mass"])
            df["Gene name"] = gene
            all_genes.append(df)

        summary = pd.concat(all_genes)
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        summary = summary[cols] 
        return summary

             
        
class PeptideCandidates():
                                      
    def __init__(self, masses, cor_features, cor_ion_images):
        self.masses = masses
        self.cor_features = cor_features
        self.cor_ion_images = cor_ion_images
        self.cor_with_other = None
        
        self.matches = None
        self.multiple_matches = None
        self.mass_shifts = None
        
    def find_correlations_between_ion_images(self, images):
        cor_with_other = np.zeros((len(self.cor_features), len(self.cor_features)))

        for i, ion in enumerate(self.cor_features):
            for j, other_ion in enumerate(self.cor_features):
                cor = np.zeros(len(images))

                for s in range(len(images)):
                    ion_image = images[s][...,ion]
                    other_ion_image = images[s][...,other_ion]
                    cm = scipy.stats.spearmanr(ion_image.flat, other_ion_image.flat)
                    cor[s] = cm.correlation

                cor_with_other[i][j] = np.mean(cor)

        self.cor_with_other = cor_with_other
        return self
        
        
    def match(self, all_msi_masses):
        matches = {}
        multiple_matches = {}

        for msi_mass in all_msi_masses[self.cor_features]: 
            genes = self.masses.loc[self.masses["MSI mass"] == msi_mass]["Gene name"].drop_duplicates()
            proteins = self.masses.loc[self.masses["MSI mass"] == msi_mass]["Protein names"].drop_duplicates()

            for gene, protein in zip(genes.items(), proteins.items()):
                index, gene = gene
                index, protein = protein
                key = gene + "__" + str(protein)
                if key not in matches: matches[key] = list()
                matches[key].append(msi_mass)

        for gene_protein, values in matches.items():
            if len(values) > 1:
                indices = []
                for v in values:
                    indices.append(np.argmin(np.abs(all_msi_masses[:] - v)))
                cor_indices = np.where(np.isin(self.cor_features, indices))[0]
                pairs = []
                for idx, cor_i in enumerate(cor_indices):
                    rel_cor = self.cor_with_other[cor_i, cor_indices[idx:]]
                    match = np.where(np.logical_and(rel_cor < 1, rel_cor > self.cor_ion_images))[0]
                    for m in match:
                        pairs.append([cor_i, cor_indices[m+idx]])

                for idx, p in enumerate(pairs):
                    if gene_protein not in multiple_matches: multiple_matches[gene_protein] = list()
                    multiple_matches[gene_protein].append((all_msi_masses[self.cor_features[p[0]]], all_msi_masses[self.cor_features[p[1]]]))

        self.matches = matches
        self.multiple_matches = multiple_matches
        return self
    

    def filter_potential_mass_shifts(self, mass_shift = 0.0487): 
        mass_shifts = {}
        true_multiple_matches = {}

        for gene, values in self.multiple_matches.items():
            for v in values:
                diff = np.abs(v[0]-v[1])
                if diff > mass_shift:
                    if gene not in true_multiple_matches: true_multiple_matches[gene] = list()
                    true_multiple_matches[gene].append(v)
                elif np.logical_and(diff < mass_shift, diff > 0.):
                    if gene not in mass_shifts: mass_shifts[gene] = list()
                    mass_shifts[gene].append(v)

        self.multiple_matches = true_multiple_matches
        self.mass_shifts = mass_shifts
        return self
    
    
    def print_results(self):
        print("Peptides for which more than one mass was assigned")
        print(len(self.multiple_matches))
        print(self.multiple_matches.keys()) 
        print("Filtered peptides for which only potential mass shifts were detected")
        print(len(self.mass_shifts))

