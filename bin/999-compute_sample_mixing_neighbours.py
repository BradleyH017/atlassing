########### Bradley
########### 
import scanpy as sc
import scipy as sp
import pandas as pd
import numpy as np
import scib_metrics
from scib_metrics import kbet
import seaborn as sns
import matplotlib.pyplot as plt
import scib as scib

# Load data
fpath = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/v3_all/v3_results_benching_batch_scANVI_290424/rectum/objects/adata_PCAd_batched_umap.h5ad"
#fpath = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/alternative_results/bi_nmads_results/rectum/objects/adata_PCAd_batched.h5ad"
adata = sc.read_h5ad(fpath)
embedding_key="Harmony"
samp_col = "samp_tissue"
k_vals = [15, 30, 50, 100, 150, 200, 250, 300, 350]
n_pcs = 14

# for each value of k:
samp_props = []
for k in k_vals:
    print(k)
    # Compute nieghbours
    print("Calculating neighbours")
    sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs, use_rep="X_" + embedding_key, key_added= "X_" + embedding_key + "_" + str(k) + "_nn")
    # Extract the indices of nearest neighbors for each cell
    neighbors_matrix = adata.obsp["X_" + embedding_key + "_" + str(k) + "_nn" + '_connectivities']
    # Extract sample identifier
    sample_ids = adata.obs[samp_col]
    # Initialize a dictionary to hold the count of cells from each sample per neighbor group
    sample_mixing = {}
    print("Calculating sample mixing")
    for i in range(neighbors_matrix.shape[0]):
        # Get indices of neighbors for the i-th cell
        neighbor_indices = neighbors_matrix[i].nonzero()[1]
        # Get the sample ids of these neighbors
        neighbor_samples = sample_ids.iloc[neighbor_indices]
        # Get the ref sample
        ref_sample = sample_ids[i]
        # Count the number of times cells from the same sample are observed within the nearest neighbours
        same_sample_count = sum(neighbor_samples == ref_sample)
        # Store the result
        sample_mixing[i] = same_sample_count
    # Convert to DataFrame for easier manipulation and visualization
    sample_mixing_df = pd.DataFrame(list(sample_mixing.items()), columns=['cell', 'Nsame_samp'])
    sample_mixing_df['cell'] = adata.obs.index
    params_dict = adata.uns["X_" + embedding_key + "_" + str(k) + "_nn"]['params']
    n_neighbours = params_dict['n_neighbors']
    sample_mixing_df[str(k) + '_prop_same_samp'] = sample_mixing_df['Nsame_samp'] / k
    samp_props.append(sample_mixing_df)

merged_df = samp_props[0][['cell', str(k_vals[0]) + '_prop_same_samp']]
for i, df in enumerate(samp_props[1:]):  # start=1 because we already used the first DataFrame
    # Select only the 'cell' and the newly renamed 'prop_same_samp' columns
    df = df.filter(regex='^(cell|.*_prop_same_samp)$')
    # Merge with the growing merged_df on 'cell'
    merged_df = pd.merge(merged_df, df, on='cell', how='outer')


# Plot the distribution of these values
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for i,k in enumerate(k_vals):
    data = merged_df[f"{str(k)}_prop_same_samp"]
    mean = sum(data) / data.shape[0]
    line_color = sns.color_palette("tab10")[i]
    sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{str(k)}, mean={mean:.2f}, max={max(data):.2f}')
    plt.axvline(x = mean, linestyle = '--', color = line_color, alpha = 0.5)

plt.legend()
plt.xlabel('Proportion of cells in neighbourhood from same sample')
plt.title(f"Same-sample bias across neighbourhood size")
plt.savefig("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/temp/nHood_sample_inclusion.png", bbox_inches='tight')
plt.clf()

# Same but log
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for i,k in enumerate(k_vals):
    data = merged_df[f"{str(k)}_prop_same_samp"]
    data_log_perc = np.log10(data*100)
    mean = sum(data_log_perc) / data.shape[0]
    line_color = sns.color_palette("tab10")[i]
    sns.distplot(data_log_perc, hist=False, rug=True, color=line_color, label=f'{str(k)}, mean={mean:.2f}, max={max(data):.2f}')
    plt.axvline(x = mean, linestyle = '--', color = line_color, alpha = 0.5)

plt.legend()
plt.xlabel('Proportion of cells in neighbourhood from same sample (log10 % same sample)')
plt.title(f"Same-sample bias across neighbourhood size")
plt.savefig("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/temp/nHood_sample_inclusion_log.png", bbox_inches='tight')
plt.clf()


# Select the minimum NN at which for ALL neighbourhoods the majority of cells are combing from alternative neighbourhoods
# i.e the first kNN value at which max proportion of same sample neighbourhoods < 0.5
knn_columns = merged_df.filter(regex='_prop_same_samp$')
max_values = knn_columns.max(axis=0)
max_df = pd.DataFrame({'kNN': [col[:-len('_prop_same_samp')] for col in knn_columns.columns], 'max_same_samp_prop': max_values})
max_df['kNN'] = pd.to_numeric(max_df['kNN'])
max_df = max_df.sort_values(by='kNN', ascending=True)

# Find min kNN with max_knn_value < threshold
threshold = 0.5
best_knn = max_df[max_df['max_same_samp_prop'] < threshold].kNN[0]

# Also calculate the inverse local simpsons index and kBet per neighbourhood
# This is a measure of the diversity of cells within a given neighbourhood
# https://www.nature.com/articles/s41592-019-0619-0
max_df['ilisi'] = 0
# Replace names for the largest knn
maxk = max(max_df['kNN'])
adata.uns['neighbors'] = adata.uns[f"X_Harmony_{str(maxk)}_nn"]
adata.obsp['connectivities'] = adata.obsp[f"X_Harmony_{str(maxk)}_nn_connectivities"] 
adata.obsp['distances'] = adata.obsp[f"X_Harmony_{str(maxk)}_nn_distances"] 

for k in k_vals:
    print(k)
    ilisi = scib.metrics.ilisi_graph(adata, batch_key=samp_col, type_="knn", k0=k, scale=True)
    max_df.loc[f"{str(k)}_prop_same_samp","ilisi"] = ilisi