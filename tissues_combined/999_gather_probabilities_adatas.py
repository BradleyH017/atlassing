# Brad Jan 2024: Gathering the adata's and keras probabilities of each cell being each annotation by keras
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import gseapy as gp
from scipy import sparse


tissues = ["blood", "ti", "rectum"]
gutbasedir = ["/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/2023_08_07", "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/2023_09_27-ccf_otar_pilots"]
blood_files = ["/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/2023_08_07/blood/results/celltype/adata_no_keras.h5ad", "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/blood/results/celltype/adata.h5ad"]
outdir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined"
annot__file = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/highQC_TI_discovery/data-clean_annotation_full.csv"
inputdir = f"{outdir}/input"
if os.path.exists(inputdir) == False:
    os.mkdir(inputdir)

# Load in the data for each tissue (across each directory)
gutdata = []
gut_exp = []
for t in tissues[1:]:
    for dir in gutbasedir:
        print(f"{t} - {dir}")
        tissue_data = sc.read_h5ad(f"{dir}/{t}/results/celltype/adata.h5ad")
        # Add the input file that was used:
        tissue_data.obs['input'] = f"{dir}/{t}/results/celltype/adata.h5ad"
        tissue_data.obs['tissue'] = t
        gutdata.append(tissue_data)
        tissue_samples = np.unique(tissue_data.obs['experiment_id'])
        gut_exp.append(tissue_samples)
        tissue_samples = np.unique(tissue_data.obs['convoluted_samplename'])
        gut_convoluted_samplename = tissue_samples
        del tissue_data


# print shapes
for i in range(0, 4): 
    print(f"adata shape: {gutdata[i].shape}")
    print(f"nsamples: {len(gut_exp[i])}")
    
# Make sure within site, there are no replicates (if there are, take the those from the second directory) [NOTE - Need to double check this is the right thing to do]
if (len(np.intersect1d(gut_exp[0], gut_exp[1])) == 0) != True:
    remove = np.intersect1d(gut_exp[0], gut_exp[1])
    gutdata[0] = gutdata[0][~gutdata[0].obs['experiment_id'].isin(remove)]
    gut_exp[0] = gut_exp[0][~np.isin(gut_exp[0], remove)]


# Check rectum (TRUE)
len(np.intersect1d(gut_exp[2], gut_exp[3])) == 0

# Now read in the blood
blooddata = []
blood_exp = []
for f in blood_files:
    print(f"blood - {f}")
    tissue_data = sc.read_h5ad(f)
    tissue_data.obs['input'] = f
    tissue_data.obs['tissue'] = 'blood'
    blooddata.append(tissue_data)
    tissue_samples = np.unique(tissue_data.obs['experiment_id'])
    blood_exp.append(tissue_samples)


# Check shapes
for i in range(0, 2): 
    print(f"adata shape: {blooddata[i].shape}")
    print(f"nsamples: {len(blood_exp[i])}")


# Check sample overlap, take keep thos in the first file (this is newer), so remove from second
if (len(np.intersect1d(blood_exp[0], blood_exp[1])) == 0) != True:
    remove = np.intersect1d(blood_exp[0], blood_exp[1])
    blooddata[1] = blooddata[1][~blooddata[1].obs['experiment_id'].isin(remove)]
    blood_exp[1] = blood_exp[1][~np.isin(blood_exp[1], remove)]


# Put all together
blood = ad.concat(blooddata)
blood.obs['label__machine_probability'] = np.nan
blood.obs['label__machine'] = ""
TI = ad.concat(gutdata[:2])
TI.obs = TI.obs.rename(columns = {'Keras:predicted_celltype_probability': 'label__machine_probability', 'Keras:predicted_celltype': 'label__machine'})
rectum = ad.concat(gutdata[2:4])
rectum.obs = rectum.obs.rename(columns = {'Keras:predicted_celltype_probability': 'label__machine_probability'})
rectum.obs = rectum.obs.rename(columns = {'Keras:predicted_celltype': 'label__machine'})
adata = ad.concat([blood, TI, rectum])
adata.var = gutdata[0].var.iloc[:,:6]

# Compute average expresssion of Kong et al 2022 gene lists: https://www.sciencedirect.com/science/article/pii/S1074761323000122?via%3Dihub#bib24 ('Cell type identification and signatures)
Kong_sets = {
    'Kong_epi': ["EPCAM", "KRT8", "KRT18"],
    'Kong_strom': ["CDH5", "COL1A1", "COL1A2", "COL6A2", "VWF"],
    'Kong_immune':["PTPRC", "CD3D", "CD3G", "CD3E", "CD79A", "CD79B", "CD14", "FCGR3A", "CD68", "CD83", "CSF1R", "FCER1G"]  # CD16 renamed in our data compared to Kong to FCGR3A (and CD45 = PTPRC in our data)
}

# Normalise data
adata.layers['counts'] = adata.X
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['lognorm'] = adata.X

# Compute mean/cell
temp = adata.X
temp=pd.DataFrame.sparse.from_spmatrix(temp)
temp.columns = adata.var['gene_symbols']
mask = temp.columns.isin(np.concatenate(list(Kong_sets.values())))
# Subset the DataFrame based on the boolean mask
temp = temp.iloc[:,mask]

# Compute averages and save
resdf = pd.DataFrame({'cell': adata.obs.index})
for gene_set_name, gene_set_genes in Kong_sets.items():
    print(f"Computing average for: {gene_set_name}")
    gene_set_subset = temp.iloc[:,temp.columns.isin(gene_set_genes)]
    resdf[f"{gene_set_name}_average"] = gene_set_subset.sum(axis=1) / len(gene_set_genes)


# Add annotation for which is the highest score per cell
resdf['Kong_highest_average'] = resdf.filter(like='_average').idxmax(axis=1)

# Now load all of the probabilities for the gut data
probs = []
for set in range(0,4):
    if set < 2:
        tissue = "ti"
    else:
        tissue = "rectum"
    if np.isin(set, [0, 2]):
        dir = gutbasedir[0]
    else:
        dir = gutbasedir[1]
    # Load in the data per sample
    print(f"{tissue} - {dir}")
    samples = np.array([element.replace("__donor", "") for element in gut_exp[set]])
    data = []
    for s in samples:
        print(s)
        temp = ad.read_h5ad(f"{dir}/{tissue}/results/celltype/keras_celltype/{s}/{s}___cellbender_fpr0pt1-scrublet-ti_freeze003_prediction-predictions.h5ad")
        want = temp.obs.columns.str.contains("probability__")
        temp_probs = temp.obs.loc[:,want]
        data.append(temp_probs)
        del temp, temp_probs

    res = pd.concat(data)
    # Now add to the probs of the rest
    probs.append(res)


keras = pd.concat(probs)
keras = keras[keras.index.isin(adata.obs.index)]
keras.to_csv(inputdir + "/keras_all_annot_probabilities.csv")

# Merge with annotation file
annot = pd.read_csv(annot__file)
annot['label__machine'] = annot['label__machine_retired']
adata.obs['cell'] = adata.obs.index
adata.obs = adata.obs.merge(annot[['category__machine', 'label__machine']], on='label__machine', how ='left')
adata.obs.set_index('cell', inplace=True)

# Make sure there are no duplicates !!!
duplicated_mask = adata.obs.index.duplicated(keep='first')
adata = adata[~duplicated_mask]

# Save the final object
category_columns = adata.obs.select_dtypes(include='category').columns
adata.obs[category_columns] = adata.obs[category_columns].astype(str)
adata.obs.drop("Azimuth:predicted.celltype.l2.score", axis=1, inplace=True)
adata.obs.drop("Azimuth:mapping.score", axis=1, inplace=True)
adata.write(inputdir + "/adata_raw_input.h5ad")


    


# Load the probabilities for the gutdata only
# Ultimate goal is to have 
data = []
for i in range(0,len(samples)):
    print(samples[i])
    temp = ad.read_h5ad(dir + "/" + samples[i] + "/" + samples[i] + "___cellbender_fpr0pt1-scrublet-ti_freeze003_prediction-predictions.h5ad")
    want = temp.obs.columns.str.contains("probability__")
    temp_probs = temp.obs.loc[:,want]
    data.append(temp_probs)
    del temp, temp_probs
