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