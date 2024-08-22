# Bradley August 2022
from anndata.experimental import read_elem
from h5py import File
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scanpy as sc
from matplotlib.patches import Wedge, Patch
from adjustText import adjust_text
from bokeh.palettes import Category10_10  


# Define outdir
outdir="other_paper_figs"

# Load in obs only
f="results/combined/objects/celltypist_prediction_conf_gt_0.5.h5ad"
f2 = File(f, 'r')
obs = read_elem(f2['obs'])

######## 1. Plot distirbution of cells per sample, for each tissue x disease #########
td = np.unique(obs['tissue_disease'])
cells_sample = pd.DataFrame(obs['sanger_sample_id'].value_counts()).reset_index()
to_add = obs[['sanger_sample_id', 'tissue_disease']].reset_index(drop=True).drop_duplicates()
cells_sample = cells_sample.merge(to_add, on="sanger_sample_id", how="left")

plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for t in td:
    print(t)
    data = cells_sample.loc[cells_sample['tissue_disease'] == t, "count"].values
    med = np.median(data)
    sns.histplot(data, stat='frequency', kde=True, label=f"{t}: {med}")

plt.legend()
plt.xlabel("Cells / sample")
plt.title(f"Number of confidently annotated cells per sample")
plt.savefig(f"{outdir}/predicted_cells_per_sample.png", bbox_inches='tight')
plt.clf()

######### 2.  Plot the proportion of tissues per celltype, compared with nCells and nSamples #########
df_grouped = obs.groupby('predicted_labels').agg(
    nCells=('predicted_labels', 'size'),
    nSamples=('sanger_sample_id', 'nunique')
).reset_index()

# Apply log10 transformation to the row count
df_grouped['log10_nCells'] = np.log10(df_grouped['nCells'])

# Calculate the proportions of tissue types within each predicted label
df_proportions = obs.groupby(['predicted_labels', 'tissue']).size().reset_index(name='count')
df_proportions['proportion'] = df_proportions.groupby('predicted_labels')['count'].transform(lambda x: x / x.sum())

# Use Bokeh's Category10 palette for tissues
tissue_types = obs['tissue'].cat.categories
tissue_colors = dict(zip(tissue_types, Category10_10[:len(tissue_types)]))  # Map tissues to Category10 colors

# Function to generate pie chart markers for scatter plot
def pie_marker(proportions, n_slices=10):
    start_angle = 0
    markers = []
    sizes = []
    for proportion in proportions:
        end_angle = start_angle + proportion * 2 * np.pi
        # Create the slice
        x = [0] + np.cos(np.linspace(start_angle, end_angle, n_slices)).tolist()
        y = [0] + np.sin(np.linspace(start_angle, end_angle, n_slices)).tolist()
        # Store as marker
        markers.append(np.column_stack([x, y]))
        sizes.append(np.abs([x, y]).max())
        start_angle = end_angle
    return markers, sizes

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# To store the text objects for repelling
texts = []

# Iterate over each group and plot pie chart markers at the corresponding point
for i, row in df_grouped.iterrows():
    # Get the proportions for this predicted_label
    label_proportions = df_proportions[df_proportions['predicted_labels'] == row['predicted_labels']]
    proportions = label_proportions['proportion'].values
    tissues = label_proportions['tissue'].values
    # Get the pie chart markers and sizes
    markers, sizes = pie_marker(proportions)
    # Coordinates for the scatter point
    x = row['log10_nCells']
    y = row['nSamples']
    # Plot each slice of the pie as a separate scatter point
    for marker, size, tissue in zip(markers, sizes, tissues):
        color = tissue_colors[tissue]  # Use the Bokeh Category10 color for the tissue
        ax.scatter(x, y, marker=marker, s=size ** 2 * 150, facecolor=color, edgecolor='black')
    # Add the text label but store it for adjustment later
    text = ax.text(x, y, row['predicted_labels'], fontsize=9, ha='center', va='center')
    texts.append(text)

# Set labels and titles
ax.set_xlabel('Log10(nCells)')
ax.set_ylabel('Number of contributing samples')

# Create legend for tissue colors using Bokeh's Category10 colors
legend_elements = [Patch(facecolor=color, edgecolor='black', label=tissue) for tissue, color in tissue_colors.items()]
ax.legend(handles=legend_elements, title="Tissues", loc="upper right", bbox_to_anchor=(1.2, 1))

# Repel the text labels using adjustText
adjust_text(texts, ax=ax, expand_text=(1.05, 1.2), expand_points=(1.2, 1.5), force_points=0.3)

plt.title('Contribution of cells, samples, and tissues to each cell-type')
plt.tight_layout()
plt.savefig(f"{outdir}/prop_tissues_per_cluster.png", bbox_inches='tight')
plt.show()

###### 3. Plot distribution of max contribution of a single tissue ######
max_idx = df_proportions.groupby('predicted_labels')['proportion'].idxmax()
df_max = df_proportions.loc[max_idx].reset_index(drop=True)
df_max['manual_lineage'] = df_max["predicted_labels"].apply(lambda x: x.split('_')[0] if '_' in x else x)
lins = np.unique(df_max['manual_lineage'])
perc20 = df_max['proportion'].quantile(0.2)
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for t in lins:
    print(t)
    data = df_max.loc[df_max['manual_lineage'] == t, "proportion"].values
    sns.distplot(data, hist=False, rug=True, label=t)

plt.axvline(x = perc20, color = 'black', linestyle = '--', alpha = 0.5)
ax.set_xlim([0,1.0])
plt.legend()
plt.xlabel("Maximum contribution by single tissue")
plt.title(f"Exclusivity of clusters to a single tissue")
plt.savefig(f"{outdir}/max_prop_single_tissue.png", bbox_inches='tight')
plt.clf()


###### 4. Compare predicted cell proportion vs high qc cell proportion ######
hqc = "results_round3/combined/objects/adata_PCAd.h5ad"
f2 = File(hqc, 'r')
hqc_obs = read_elem(f2['obs'])

pred_props = pd.DataFrame(obs['predicted_labels'].value_counts(normalize=True)).reset_index(names="leiden")
pred_props.rename(columns={"proportion": "prediction_proportion"}, inplace=True)
hqc_props = pd.DataFrame(hqc_obs['leiden'].value_counts(normalize=True)).reset_index(names="leiden")
hqc_props.rename(columns={"proportion": "high_qc_proportion"}, inplace=True)
prop_plt = pred_props.merge(hqc_props, on="leiden", how="left")
prop_plt['manual_lineage'] = prop_plt["leiden"].apply(lambda x: x.split('_')[0] if '_' in x else x)

fig, ax = plt.subplots(figsize=(8, 6))
for lineage in prop_plt['manual_lineage'].unique():
    subset = prop_plt[prop_plt['manual_lineage'] == lineage]
    ax.scatter(subset['high_qc_proportion'], subset['prediction_proportion'],
               label=lineage)

ax.plot([0, 1], [0, 1], ls='--', color='black', label='x = y')
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 0.3)
ax.set_xlabel('High QC Proportion')
ax.set_ylabel('Prediction Proportion')
plt.legend()
ax.set_title('Concordance of high QC and predicted labels')
plt.savefig(f"{outdir}/highqc_vs_predicted_proportions.png", bbox_inches='tight')
plt.clf()

# Have a look at the biggest changers
prop_plt['pred_over_highqc'] = prop_plt['prediction_proportion'] / prop_plt['high_qc_proportion']