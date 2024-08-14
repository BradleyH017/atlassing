import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from anndata.experimental import read_elem
from h5py import File

tissues = ["rectum", "TI", "blood"]
obs = []
for t in tissues:
    print(t)
    f = f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/CELLRANGER_V7/annotated_anndata/2024_05_15-{t}_verify_annot.h5ad"
    f2 = File(f, 'r')
    # Read only cell-metadata
    x = read_elem(f2['obs'])
    print(x.shape)
    obs.append(x)

x = pd.concat(obs)

# Generate the depth_counts
x.rename(columns = {"Exp": "sanger_sample_id"}, inplace=True)
samp_data = np.unique(x['sanger_sample_id'], return_counts=True)
cells_sample = pd.DataFrame({'sanger_sample_id': samp_data[0], 'Ncells':samp_data[1]})
# Summarise
depth_count = pd.DataFrame(index = np.unique(x.sanger_sample_id), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts", "Median_nCounts", "Median_nGene_by_counts"])
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,1] = x[x.sanger_sample_id == samp].shape[0]
    depth_count.iloc[s,0] = sum(x[x.sanger_sample_id == samp].total_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,3] = sum(x[x.sanger_sample_id == samp].n_genes_by_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,4] = np.median(x[x.sanger_sample_id == samp].total_counts)
    depth_count.iloc[s,5] = np.median(x[x.sanger_sample_id == samp].n_genes_by_counts)

depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))
depth_count["log10_Median_nCounts"] = np.log10(np.array(depth_count["Median_nCounts"].values, dtype = "float"))
depth_count.rename(columns={"n_genes_by_counts": "Mean_n_genes_by_counts"}, inplace=True)
depth_count.drop(columns="High_cell_sample", inplace=True)
depth_count.reset_index(inplace=True)
depth_count.rename(columns={"index": "sanger_sample_id"}, inplace=True)

# Save this 
from datetime import datetime
today = datetime.today().strftime('%d%m%Y')
depth_count.to_csv(f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/wetlab_plots/{today}_sample_metadata_recomputed.csv", index=False)

# Check discovery/replication - TI paper
dr = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/monika_analysis/andersonlab-select_samples_atlas/ti_atlas/outputs/discovery_replication_final.tsv", sep = "\t")
dcdr = depth_count[depth_count['sanger_sample_id'].isin(dr['sanger_sample_id'])]
min(dcdr['Median_nGene_by_counts'])
# Save 
dcdr.to_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/wetlab_plots/20062024_TI_atlas_sample_QC_from_expression.csv", index=False)


# Merge with the raw metadata (originally used to subset the samples)
df = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/tobi_qtl_analysis/code/andersonlab-select_samples_atlas/metadata_combined/metadata_combined.tsv", sep = "\t")
df['sanger_sample_id'] = df['sanger_sample_id'] + "_" + df['biopsy_type_final']
df = df[df['sanger_sample_id'].isin(depth_count['sanger_sample_id'])]
df.rename(columns={"median_genes_per_cell": "metadata_ngenes_per_cel"}, inplace=True)
depth_count = depth_count.merge(df[["sanger_sample_id", "metadata_ngenes_per_cel"]], how="left", on="sanger_sample_id")


# Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='median_bh_ngenes', y='metadata_ngenes_per_cel', data=depth_count, ax=ax)
min_val = min(depth_count['metadata_ngenes_per_cel'].min(), depth_count['median_bh_ngenes'].min())
max_val = max(depth_count['metadata_ngenes_per_cel'].max(), depth_count['median_bh_ngenes'].max())
x_vals = np.linspace(min_val, max_val, 100)
y_vals = x_vals
ax.plot(x_vals, y_vals, 'r--', label='y = x')
ax.axvline(x=100, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Computed median')
ax.set_ylabel('Metadata median')
ax.set_title("Median nGenes per sample")
plt.savefig("tissues_combined/temp/meta_vs_computed_median_nGenes_per_sample.png")
plt.clf()

# Check TI
depth_count['tissue'] = depth_count['sanger_sample_id'].str.split('_').str[-1]
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='median_bh_ngenes', y='metadata_ngenes_per_cel', data=depth_count[depth_count['tissue'] == "ti"], ax=ax)
min_val = min(depth_count['metadata_ngenes_per_cel'].min(), depth_count['median_bh_ngenes'].min())
max_val = max(depth_count['metadata_ngenes_per_cel'].max(), depth_count['median_bh_ngenes'].max())
x_vals = np.linspace(min_val, max_val, 100)
y_vals = x_vals
ax.plot(x_vals, y_vals, 'r--', label='y = x')
ax.axvline(x=500, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Computed median')
ax.set_ylabel('Metadata median')
ax.set_title("Median nGenes per sample (TI)")
plt.savefig("tissues_combined/temp/meta_vs_computed_median_nGenes_per_sample_ti.png")
plt.clf()

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='median_bh_ngenes', y='Median_nGene_by_counts', data=depth_count, ax=ax)
min_val = min(depth_count['Median_nGene_by_counts'].min(), depth_count['median_bh_ngenes'].min())
max_val = max(depth_count['Median_nGene_by_counts'].max(), depth_count['median_bh_ngenes'].max())
x_vals = np.linspace(min_val, max_val, 100)
y_vals = x_vals
ax.plot(x_vals, y_vals, 'r--', label='y = x')
ax.axvline(x=100, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Computed median (from raw)')
ax.set_ylabel('Computed median (x)')
ax.set_title("Median nGenes per sample")
plt.savefig("tissues_combined/temp/obs_vs_computed_median_nGenes_per_sample.png")
plt.clf()

