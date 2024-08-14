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
compute_kong=False
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
blood = ad.concat(blooddata, join="outer")
blood.obs['label__machine_probability'] = np.nan
blood.obs['label__machine'] = ""
TI = ad.concat(gutdata[:2], join="outer")
TI.obs = TI.obs.rename(columns = {'Keras:predicted_celltype_probability': 'label__machine_probability', 'Keras:predicted_celltype': 'label__machine'})
rectum = ad.concat(gutdata[2:4], join="outer")
rectum.obs = rectum.obs.rename(columns = {'Keras:predicted_celltype_probability': 'label__machine_probability'})
rectum.obs = rectum.obs.rename(columns = {'Keras:predicted_celltype': 'label__machine'})
adata = ad.concat([blood, TI, rectum])
adata.var = gutdata[0].var.iloc[:,:6]
del blood, TI, rectum, gutdata, blooddata

# Add tissue to the index - may have duplications of sample / barcode for when we have correlated samples
adata.obs.index = adata.obs.index + "_" + adata.obs['tissue']
# Check these are now all unique?
adata.shape[0] == len(np.unique(adata.obs.index))

# Append the post qc genotyping ID from the gut metadata
mapping = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/2024_02_19_sanger_sample_id_to_Post_QC_Genotyping_ID.csv")
mapping = mapping.rename(columns = {'sanger_sample_id': 'convoluted_samplename'})
mapping.drop(columns=['patient_id'], inplace=True)
mapping['Post_QC_Genotyping_ID'] = mapping['Post_QC_Genotyping_ID'].astype(str).str.strip()
mapping = mapping[mapping['convoluted_samplename'].isin(adata.obs['convoluted_samplename'])]
mapping.replace('nan', np.nan, inplace=True)
mapping = mapping.dropna(subset="Post_QC_Genotyping_ID")
adata.obs = adata.obs.reset_index()
adata.obs = adata.obs.merge(mapping, on="convoluted_samplename", how="left")
adata.obs.set_index("index", inplace=True)
adata.obs['Post_QC_Genotyping_ID'] == adata.obs['Post_QC_Genotyping_ID'].astype(str)
adata.obs.replace('nan', np.nan, inplace=True)

# Add the annotations to the adata object as it is (Lineage probability is the same as the lineage of the top probability)
annot = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/highQC_TI_discovery/data-clean_annotation_full.csv")
annot['label__machine'] = annot['label__machine_retired']
adata.obs['cell'] = adata.obs.index
adata.obs = adata.obs.merge(annot[['category__machine', 'label__machine']], on='label__machine', how ='left')
adata.obs.set_index('cell', inplace=True)
adata.obs['lineage'] = np.where(adata.obs['category__machine'].isin(['B_Cell', 'B_Cell_plasma', 'T_Cell', 'Myeloid']), 'Immune', "")
adata.obs['lineage'] = np.where(adata.obs['category__machine'].isin(['Stem_cells', 'Secretory', 'Enterocyte']), 'Epithelial', adata.obs['lineage'])
adata.obs['lineage'] = np.where(adata.obs['category__machine']== 'Mesenchymal', 'Mesenchymal', adata.obs['lineage'])

# Compute kong?
if compute_kong:
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
    resdf['Kong_highest_average'] = resdf.filter(like='_average').idxmax()



# Now load all of the probabilities for the gut data - Need to make sure the cell ID not also incorporates the tissue! May be getting duplicated bar codes
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
    # Concat
    res = pd.concat(data)
    res.index = res.index + "_" + tissue
    # Now add to the probs of the rest
    probs.append(res)


keras = pd.concat(probs)
keras = keras[keras.index.isin(adata.obs.index)]
keras.to_csv(inputdir + "/keras_all_annot_probabilities_gut.csv")

# Add category annotation onto the keras probabilities
keras.columns = keras.columns.str.replace('probability__', '')
keras['label__machine'] = keras.idxmax(axis=1)
keras.reset_index(inplace=True)
# Merge this with the annot
keras = keras.merge(annot[['category__machine', 'label__machine']], on="label__machine", how="left")
# Derive lineage annotation:
lineages_categories = {"Immune": ['B_Cell', 'B_Cell_plasma', 'T_Cell', 'Myeloid'], 
             "Epithelial": ['Stem_cells', 'Secretory', 'Enterocyte'], 
             "Mesenchymal": "Mesenchymal"}

# Make this for lineages to labels
category_label_mapping = {}

for category, values in lineages_categories.items():
    if isinstance(values, list):
        category_label_mapping[category] = [label for value in values for label in annot['label__machine'][annot['category__machine'].str.contains(value)].tolist()]
    elif isinstance(values, str):
        category_label_mapping[category] = annot['label__machine'][annot['category__machine'].str.contains(values)].tolist()

# Convert the dictionary to a list
lineages_labels = [(name, np.unique(sublist).tolist()) for name, sublist in category_label_mapping.items()]
for lineage, labels in lineages_labels:
    print(lineage)
    temp = keras.iloc[:,keras.columns.isin(labels)]
    keras[f'sum_{lineage}'] = temp.sum(axis=1)

# Find majority lineage score
sum_columns = keras.filter(like='sum_')
keras['sum_majority_lineage'] = sum_columns.idxmax(axis=1)
keras['sum_majority_lineage'] = keras['sum_majority_lineage'].str.replace('sum_', '')

# Merge the desired columns with the anndata object
#duplicated_mask = keras.cell.duplicated(keep='first')
#keras = keras[~duplicated_mask]

keras = keras.rename(columns={'index': 'cell'})
adata.obs.reset_index('cell', inplace=True)
adata.obs = adata.obs.merge(keras[["sum_Immune", "sum_Epithelial", "sum_Mesenchymal", "sum_majority_lineage", "cell"]], on="cell", how="left")
adata.obs.set_index("cell", inplace=True)

# Incorporate Lucia's tissue swaps (24/04/24)
ts = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/lucia_analysis/tissue_swaps/data/metadata_predictions_final_2023.11.20.csv")
ts = ts.rename(columns={"sanger_sample_id": "convoluted_samplename"})
ts_to_add = ts[["convoluted_samplename", "biopsy_type_final"]]
adata = adata[adata.obs['convoluted_samplename'].isin(ts_to_add['convoluted_samplename'])]
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(ts_to_add, on = "convoluted_samplename")
adata.obs = adata.obs.drop(columns="tissue") # Drop old tissue
adata.obs = adata.obs.rename(columns={"biopsy_type_final": "tissue"})
adata.obs['tissue'] = adata.obs['tissue'].replace('r', 'rectum')
adata = adata[adata.obs['tissue'].notna()]
adata.obs.set_index("cell", inplace=True)


# Save the final object
category_columns = adata.obs.select_dtypes(include='category').columns
adata.obs[category_columns] = adata.obs[category_columns].astype(str)
adata.obs.drop("Azimuth:predicted.celltype.l2.score", axis=1, inplace=True)
adata.obs.drop("Azimuth:mapping.score", axis=1, inplace=True)
adata.obs.drop("patient_number", axis=1, inplace=True)
# CHECK THIS STILL HAS GENE SYMBOLS IN THE ADATA.VAR

# Make sure there are no non-CD blood samples
mask = (adata.obs['disease_status'] == "healthy") & (adata.obs['tissue'] == "blood")
invert_mask = ~mask
adata = adata[invert_mask, :]

# Make sure there are not individuals aged 70 or more
adata = adata[adata.obs['age'] < 70]

# Finally save
adata.write(inputdir + "/adata_raw_input_all.h5ad")

# Also save this per tissue
tissues = np.unique(adata.obs['tissue'].astype(str))
for t in tissues:
    print(t)
    temp = adata[adata.obs['tissue'] == t]
    temp.write_h5ad(f"{inputdir}/adata_raw_input_{t}.h5ad")

# And save for gut
adata.obs['tissue'] = adata.obs['tissue'].astype(str)
gut = adata[adata.obs['tissue'].isin(["ti", "rectum"])]
gut.write_h5ad(f"{inputdir}/adata_raw_input_gut.h5ad")

# Also save a test set per tissue
for t in tissues:
    print(t)
    temp = adata[adata.obs['tissue'] == t]
    sc.pp.subsample(temp, 0.02)
    temp.write_h5ad(f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/input_test/adata_raw_input_{t}.h5ad")


changed_lineage = adata.obs[adata.obs['lineage'].astype(str) != adata.obs['sum_majority_lineage'].astype(str)]
changed_lineage.to_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/input_test/max_ct_not_e_sum_majority_lineage.csv")