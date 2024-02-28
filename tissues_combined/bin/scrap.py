import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd

blood_file = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/blood/results/merged_h5ad/outlier_filtered_adata.h5ad"
blood = sc.read_h5ad(blood_file)

var="pct_counts_gene_group__mito_transcript"
df = blood.obs

# Write a function to calculate the MAD and nMAD of cells for a desired metric (one at a time)
def nmad_calc(data):
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    # Use direction aware filtering
    nmads = (data - np.median(data))/mad
    return(nmads)

def nmad_append(df, var, group=[]):
    # Calculate
    if group:
        vals = df.groupby(group)[var].apply(nmad_calc)
        # Return to df
        temp = pd.DataFrame(vals)
        temp.reset_index(inplace=True)
        if "level_1" in temp.columns:
            temp.set_index("level_1", inplace=True)
        else: 
            temp.set_index("cell", inplace=True)
        
        temp = temp.reindex(df.index)
        return(temp[var])
    else:
        vals = nmad_calc(df[var])
        return(vals)

blood.obs['test_within_keras'] = nmad_append(blood.obs, var, group="Keras:predicted_celltype")
blood.obs['test_across_all'] = nmad_append(blood.obs, var, group=False)

# Randomly sploit into TI or blood
adata = blood
annot = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/highQC_TI_discovery/data-clean_annotation_full.csv")
annot['label__machine'] = annot['label__machine_retired']
adata.obs = adata.obs.rename(columns = {'Keras:predicted_celltype': 'label__machine'})
adata.obs['cell'] = adata.obs.index
adata.obs = adata.obs.merge(annot[['category__machine', 'label__machine']], on='label__machine', how ='left')
adata.obs.set_index('cell', inplace=True)
adata.obs['lineage'] = np.where(adata.obs['category__machine'].isin(['B_Cell', 'B_Cell_plasma', 'T_Cell', 'Myeloid']), 'Immune', "")
adata.obs['lineage'] = np.where(adata.obs['category__machine'].isin(['Stem_cells', 'Secretory', 'Enterocyte']), 'Epithelial', adata.obs['lineage'])
adata.obs['lineage'] = np.where(adata.obs['category__machine']== 'Mesenchymal', 'Mesenchymal', adata.obs['lineage'])
adata.obs['tissue'] = np.random.choice(['TI', 'blood'], size=adata.obs.shape[0])

# Get relative MT% within lineage
adata.obs['nmad_mito_transcript_tissue_lineage'] = nmad_append(adata.obs, 'pct_counts_gene_group__mito_transcript', group=['tissue', 'lineage'])
# Compare with not within tissue
adata.obs['nmad_mito_transcript_lineage'] = nmad_append(adata.obs, 'pct_counts_gene_group__mito_transcript', group='lineage')
adata.obs['tissue'] = np.random.choice(['TI', 'blood'], size=adata.obs.shape[0])

# Do  the relative thresholds on depth counts
adata.obs['samp_tissue'] = adata.obs['experiment_id'].astype('str') + "_" + adata.obs['tissue'].astype('str')
samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
depth_count = pd.DataFrame(index = np.unique(adata.obs.samp_tissue), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts"])    
high_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
low_samps = np.array(cells_sample.loc[cells_sample.Ncells < 500, "sample"])
# Summarise
depth_count = pd.DataFrame(index = np.unique(adata.obs.samp_tissue), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts"])
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,1] = adata.obs[adata.obs.samp_tissue == samp].shape[0]
    depth_count.iloc[s,0] = sum(adata.obs[adata.obs.samp_tissue == samp].total_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,3] = sum(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)/depth_count.iloc[s,1]
    if samp in high_samps:
        depth_count.iloc[s,2] = "Red"
    else: 
        depth_count.iloc[s,2] = "Navy"
    # Also annotate the samples with low number of cells - Are these sequenced very deeply?
    if samp in low_samps:
        depth_count.iloc[s,2] = "Green"

depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))


sample_tissue = adata.obs[['samp_tissue','tissue']].reset_index()
sample_tissue = sample_tissue[['samp_tissue','tissue']].drop_duplicates()
depth_count.reset_index(inplace=True)
depth_count = depth_count.rename(columns = {"index": "samp_tissue"})
depth_count = depth_count.merge(sample_tissue, on="samp_tissue")
depth_count.set_index("samp_tissue", inplace=True)

# Find relative cutoffs per tissue
nMads=depth_count.groupby('tissue')[['nCells', 'n_genes_by_counts']].apply(nmad_calc)
unique_tissues = nMads.index.get_level_values('tissue').unique()
result_list = []
for tissue in unique_tissues:
    tissue_data = pd.DataFrame(nMads.loc[tissue])
    tissue_data['tissue'] = tissue
    result_list.append(tissue_data)

result_df = pd.concat(result_list)
result_df = result_df.reset_index()
result_df = result_df.rename(columns = {'Mean_nCounts': 'Mean_nCounts_nMad', 'n_genes_by_counts': 'n_genes_by_counts_nMad'})
depth_count = depth_count.merge(result_df[['samp_tissue', 'Mean_nCounts_nMad', 'n_genes_by_counts_nMad']], on="samp_tissue")
# Exclude those on the lower end only
depth_count['samp_nCells_keep'] = depth_count['Mean_nCounts_nMad'] > -(relative_nMAD_threshold)
depth_count['samp_n_genes_by_counts_keep'] = depth_count['n_genes_by_counts_nMad'] > -(relative_nMAD_threshold)
depth_count['keep_both'] = depth_count['samp_nCells_keep'] & depth_count['samp_n_genes_by_counts_keep']
adata.obs['samples_keep'] = adata.obs['samp_tissue'].isin(depth_count[depth_count['keep_both'] == True]['samp_tissue'])