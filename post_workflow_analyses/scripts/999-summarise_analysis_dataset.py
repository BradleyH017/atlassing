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

# Get the colormap used by Scanpy's `sc.pl.umap`
# Assuming that `obs` has `tissue` as a categorical variable with its palette
sc.pl.umap(obs, color='tissue', show=False)  # This forces scanpy to load the colors
tissue_colors = dict(zip(obs['tissue'].cat.categories, obs.uns['tissue_colors']))

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
        color = tissue_colors[tissue]  # Use Scanpy's UMAP color for the tissue
        ax.scatter(x, y, marker=marker, s=size ** 2 * 150, facecolor=color, edgecolor='black')
    # Add the text label but store it for adjustment later
    text = ax.text(x, y, row['predicted_labels'], fontsize=9, ha='center', va='center')
    texts.append(text)

# Set labels and titles
ax.set_xlabel('Log10(nCells)')
ax.set_ylabel('Number of contributing samples')

# Create legend for tissue colors using Scanpy's tissue colors
legend_elements = [Patch(facecolor=color, edgecolor='black', label=tissue) for tissue, color in tissue_colors.items()]
ax.legend(handles=legend_elements, title="Tissues", loc="upper right", bbox_to_anchor=(1.2, 1))

# Repel the text labels using adjustText
adjust_text(texts, ax=ax, expand_text=(1.05, 1.2), expand_points=(1.2, 1.5), force_points=0.3)

plt.title('Contribution of cells, samples, and tissues to each cell-type')
plt.tight_layout()
plt.savefig(f"{outdir}/prop_tissues_per_cluster.png", bbox_inches='tight')
plt.show()