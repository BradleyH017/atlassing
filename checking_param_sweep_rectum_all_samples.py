##### Bradley September 2023
##### Checking the atlassing of the rectum
##### conda activate sc4

# Load in the libraries
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
import kneed as kd
import scvi
import sys
import csv
import datetime
import seaborn as sns
import re
import scipy.stats as st
#sys.path.append('/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/conda_envs/scRNAseq/CellRegMap')
#from cellregmap import run_association, run_interaction, estimate_betas
print("Loaded libraries")

# Changedir
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
cwd = os.getcwd()
print(cwd)

# Define variables and options
n=350
min_dist=0.5
spread=0.5
data_name="rectum"
status="healthy"
category="All_not_gt_subset"
param_sweep_path = data_name + "/" + status + "/" + category + "/objects/adata_objs_param_sweep"
statpath = "rectum/" + status
catpath = statpath + "/" + category
figpath = catpath + "/figures"
tabdir = catpath + "/tables"

# Load batched file
objpath = data_name + "/" + status + "/" + category + "/objects"
batched_file = objpath + "/adata_PCAd_batched.h5ad"
adata = ad.read_h5ad(batched_file)

# Summary of the best integration method
metrics = pd.read_csv(tabdir + "/integration_benchmarking.csv", index_col=0)
metrics_plot = metrics.iloc[:-1,].astype(float)

# Plot these
plt.figure(figsize=(14, 6))  # Set the figure size
sns.heatmap(metrics_plot, annot=True, cmap='coolwarm', linewidths=0.5)
plt.axvline(x=5, color='black', linewidth=3)
plt.axvline(x=10, color='black', linewidth=3)
plt.axvline(x=12, color='black', linewidth=3)
plt.title('Summary of integration metrics')
plt.xlabel('Integration metric')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=90)
plt.savefig(figpath + '/Integration_metric_summary.png', bbox_inches='tight')
plt.clf()

metric_types = ["Bio conservation", "Batch correction"]
for m in metric_types:
    print(m)
    temp = metrics.T
    subset_df = temp[temp['Metric Type'] == m]
    subset_means = np.mean(subset_df.iloc[:,:-1].astype(float), axis=0)
    print(subset_means)

# Also have a look at if we remove the bad annotations
metrics = pd.read_csv(tabdir + "/integration_benchmarking_confident_annotations_only.csv", index_col=0)
metrics_plot = metrics.iloc[:-1,].astype(float)

# Plot these
plt.figure(figsize=(14, 6))  # Set the figure size
sns.heatmap(metrics_plot, annot=True, cmap='coolwarm', linewidths=0.5)
plt.axvline(x=5, color='black', linewidth=3)
plt.axvline(x=10, color='black', linewidth=3)
plt.axvline(x=12, color='black', linewidth=3)
plt.title('Summary of integration metrics - Confidence annotations only')
plt.xlabel('Integration metric')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=90)
plt.savefig(figpath + '/Integration_metric_summary_conf_only.png', bbox_inches='tight')
plt.clf()


# Define fig out dir
sc.settings.figdir=data_name + "/" + status + "/" + category + "/figures"
plt.gcf().set_dpi(500)

# Decide latent variable
latent_use="X_scVI"
adata.obsm['X_umap'] = adata.obsm['UMAP_' + latent_use]

# Load in Elmentaite majority voting
statpath = data_name + "/" + status
catpath = statpath + "/" + category
figpath = catpath + "/figures"
tabdir = catpath + "/tables"
intestinal_annot = pd.read_csv(tabdir + "/CellTypist_labels.csv", index_col=0)
intestinal_annot = intestinal_annot.add_prefix('CellTypist:Intestinal_Elmentaite:')
adata.obs.index = adata.obs.index.astype(str)
cells = adata.obs.index
adata.obs = adata.obs.merge(intestinal_annot, left_index=True, right_index=True, how='left')

# Plot categories, labels, samples within categories, lineages and across all
adata.obs['Keras:predicted_celltype_probability'] = adata.obs['Keras:predicted_celltype_probability'].astype("float")
adata.obs['log10ngenes'] = np.log10(adata.obs['n_genes_by_counts'])
adata.obs['log10_ncounts'] = np.log10(adata.obs.total_counts)
# Have a look at the probabilities of CellTypist annotations.
ct_probs = pd.read_csv(tabdir+ "/CellTypist_Elmentaite_prob_matrix.csv", index_col=0)
ct_probs['top_conf_Elmentaite'] = ct_probs.max(axis=1)
ct_probs = ct_probs[ct_probs.index.isin(adata.obs.index)]
adata.obs['Azimuth:predicted.celltype.l2.score'] = adata.obs['Azimuth:predicted.celltype.l2.score'].astype('float')
adata.obs['Azimuth:mapping.score'] = adata.obs['Azimuth:mapping.score'].astype('float')
adata.obs = adata.obs.merge(ct_probs['top_conf_Elmentaite'], left_index=True, right_index=True, how="left")
features = ["category", "label", "bead_lot", "gem_lot", "convoluted_samplename", "Keras:predicted_celltype_probability", "n_genes_by_counts", "log10ngenes", "log10_ncounts",
            "pct_counts_gene_group__mito_transcript","log10ngenes", "pct_counts_gene_group__ribo_protein", "total_counts", "Celltypist:Immune_All_Low:majority_voting", 
            "Celltypist:Immune_All_High:majority_voting", "CellTypist:Intestinal_Elmentaite:majority_voting", "top_conf_Elmentaite", 'Azimuth:predicted.celltype.l2', 
            'Azimuth:predicted.celltype.l2.score', 'Azimuth:mapping.score', 'Azimuth:L0_predicted.celltype.l2', 'Azimuth:L1_predicted.celltype.l2']

levels = ["all", "category", "lineage"]
cats = np.unique(adata.obs.category)
lins = np.unique(adata.obs.lineage)
for l in levels:
    ldir = data_name + "/" + status + "/" + category + "/figures/" + l
    sc.settings.figdir = ldir
    if not os.path.exists(ldir):
        os.mkdir(ldir)
    # Plot each of the features in this directory
    for f in features:
        if l == "all":
            sc.pl.umap(adata, color=f,frameon=True, save="_post_batch_post_sweep_" + f + ".png")
        else:
            resolutions = np.unique(adata.obs[l])
            for r in resolutions:
                sc.pl.umap(adata[adata.obs[l] == r], color=f, title=r, frameon=True, save="_post_batch_post_sweep_" + f + "_" + r + ".png")
            
# Also plot the distributions of QC metrics
adata.obs['log10_ncounts'] = np.log10(adata.obs.total_counts)
metrics = ["n_genes_by_counts", "log10ngenes", "pct_counts_gene_group__mito_transcript", "pct_counts_gene_group__ribo_protein", "log10_ncounts", "total_counts"]
for l in levels[1:]:
    ldir = data_name + "/" + status + "/" + category + "/figures/" + l
    for m in metrics:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        resolutions = np.unique(adata.obs[l])
        for r in resolutions:
            sns.distplot(adata.obs[adata.obs[l] == r][m], hist=False, rug=True, label=r)
        
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.title('Distribution of ' + m)
        plt.xlabel(m + ' / cell')
        plt.savefig(ldir + '/dist_' + m + '_per_cell_per_' + r + '_after_QC.png', bbox_inches='tight')
        plt.clf()



# Check some very obvious markers across these cells
markers = pd.read_csv("rectum/check_markers.csv")
helm_cats = np.unique(markers.Category)
helm_cats = np.char.mod('%s', helm_cats)
helm_cats = np.char.replace(helm_cats, '/', '-')
conv = pd.DataFrame(adata.var['gene_symbols'])
conv = conv.reset_index()
conv.columns.values[0] = "ensembl"
markers = markers.merge(conv, on="gene_symbols")
markers = pd.DataFrame(markers)
markerdir=figpath + "/marker_figs"
if not os.path.exists(markerdir):
    os.mkdir(markerdir)

sc.settings.figdir=markerdir
for h in helm_cats:
    print(h)
    for m in markers[markers.Category == h].ensembl:
        print(m)
        sc.pl.umap(adata, color=m, title=np.array_str(markers[markers.ensembl == m].gene_symbols.values), frameon=True, save="_post_batch_post_sweep_" + h + "_" + np.array_str(markers[markers.ensembl == m].gene_symbols.values) + ".png")

# Have a couple questionable clusters, so plot the categories with their anotation over the cells
for c in cats:
    l="category"
    print(c)
    ldir = data_name + "/" + status + "/" + category + "/figures/" + l
    sc.settings.figdir = ldir
    sc.pl.umap(adata[adata.obs.category == c], color="label", title=c, frameon=True, legend_loc="on data", save="_post_batch_post_sweep_label_" + c + "_legend_on.png")

# Make an additional set of markers to query
others = pd.DataFrame({"label": ["T cell CD4+ PASK+ CCR7+", "T cell CD4+ PASK+ CCR7+", "T cell CD4+ PASK+ CCR7+",
                                "Paneth cell", "Paneth cell", "Paneth cell", 
                                "Enterocyte middle villus (1)", "Enterocyte middle villus (1)", "Enterocyte middle villus (1)",
                                "ILC3", "ILC3", "ILC3", "ILC3",
                                "Endocrine cell", "Endocrine cell", "Endocrine cell"], 
                       "gene_symbols": ["ST8SIA1", "AC139720.1", "SARDH", 
                                        "LINC01819", "RASD2", "SCGB1C2",
                                        "DUOXA2", "ZNF488", "PDX1",
                                        "PAGE4", "TAS2R42", "GOLGA8M", "AL049870.2",
                                        "DSCAM", "PAX4", "ST8SIA3"]})
others = others.merge(conv, on="gene_symbols")
others = others.merge(adata.obs[['label', 'category']], on='label', how="left")
others.drop_duplicates(inplace=True)
sc.settings.figdir=markerdir
for r in range(0,others.shape[0]):
    label=others.label.values[r]
    cat=others.category.values[r]
    gene_sym = others.gene_symbols.values[r]
    ens = others.ensembl.values[r]
    sc.pl.umap(adata[adata.obs.category == cat], color=ens, title=cat + " - " + label + " " + gene_sym, frameon=True, save="_post_batch_post_sweep_" + cat + "_" + gene_sym + ".png")

# Looks like we might have found microfold cells!
# Have a look at these markers in the data
microfold = pd.DataFrame({"label": ["Microfold", "Microfold", "Microfold", "Microfold", "Microfold", "Microfold", "Microfold"], 
                       "gene_symbols": ["GP2", "TNFSF11", "SPIB", "CCL9", "CCL20", "CCL23", "TFF2"], 
                       "category": ["Enterocyte", "Enterocyte", "Enterocyte", "Enterocyte", "Enterocyte", "Enterocyte", "Enterocyte"]})
microfold = microfold.merge(conv, on="gene_symbols")
sc.settings.figdir=markerdir
for r in range(0,microfold.shape[0]):
    label=microfold.label.values[r]
    cat=microfold.category.values[r]
    gene_sym = microfold.gene_symbols.values[r]
    ens = microfold.ensembl.values[r]
    sc.pl.umap(adata[adata.obs.category == cat], color=ens, title=cat + " - " + label + " " + gene_sym, frameon=True, save="_post_batch_post_sweep_" + cat + "_" + gene_sym + ".png")

# plot secretory without the CellTypist 'TA' cells to check this is what they are being labelled as
temp = adata[adata.obs["CellTypist:Intestinal_Elmentaite:majority_voting"] != "TA"]
sc.pl.umap(temp[temp.obs.category == "Secretory"], color="CellTypist:Intestinal_Elmentaite:majority_voting", title = "Secretory - no Elmentaite TA", frameon=True, save="_post_batch_post_sweep_Secretory_CellTypist:Intestinal_Elmentaite:majority_voting_no_TA.png")

# Do these express Goblet/TA markers? (Secretory TA markers obtained from Smillie et al Supplementary Table)
goblets = pd.DataFrame({"label": ["Goblet cell middle villus", "Goblet cell middle villus", "Goblet cell middle villus", 
                                  "Secretory TA", "Secretory TA","Secretory TA"], 
                       "gene_symbols": ["CLCA1", "FCGBP", "MUC2", 
                                        "LGALS4", "TFF3", "PIGR"], 
                       "category": ["Secretory", "Secretory", "Secretory",
                                    "Secretory", "Secretory", "Secretory"]})
goblets = goblets.merge(conv, on="gene_symbols")
sc.settings.figdir=markerdir
for r in range(0,goblets.shape[0]):
    label=goblets.label.values[r]
    cat=goblets.category.values[r]
    gene_sym = goblets.gene_symbols.values[r]
    ens = goblets.ensembl.values[r]
    sc.pl.umap(adata[adata.obs.category == cat], color=ens, title=cat + " - " + label + " " + gene_sym, frameon=True, save="_post_batch_post_sweep_" + cat + "_" + gene_sym + ".png")



# Plotting nUMI
lins=np.unique(adata.obs.lineage)
adata.obs['log10_ncounts'] = np.log10(adata.obs.total_counts)
for l in lins:
    print(l)
    sc.pl.umap(adata[adata.obs.lineage == l], color='total_counts', title=l, frameon=True, save="_" + l + "_post_batch_post_sweep_nCounts.png")
    sc.pl.umap(adata[adata.obs.lineage == l], color='log10_ncounts', title=l, frameon=True, save="_" + l + "_post_batch_post_sweep_log10nCounts.png")

# Plot distribution of nCounts per lineage
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    sns.distplot(adata.obs[adata.obs.lineage == l].log10_ncounts, hist=False, rug=True, label=l)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Distribution of log10_ncounts')
plt.xlabel('MT% / cell')
#plt.axvline(x = 0.7, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/log10_ncounts_per_cell_per_lineage_after_QC.png', bbox_inches='tight')
plt.clf()

# Fit nMAd for log10_counts? RP%?
features=["pct_counts_gene_group__ribo_protein", "log10_ncounts"]
lins = np.unique(adata.obs.lineage)
for f in features:
    print(f)
    tempdata = []
    for index,c in enumerate(lins):
        print(c)
        temp = adata[adata.obs.lineage == c]
        counts=temp.obs[f]
        absolute_diff = np.abs(counts - np.median(counts))
        mad = np.median(absolute_diff)
        # Use direction aware filtering
        nmads = (counts - np.median(counts))/mad
        # Apply the function to change values in 'new_column' based on criteria
        temp.obs[f + "_nmads"] = nmads
        # Also for absolute nMADs
        nmads_abs = absolute_diff/mad
        temp.obs["abs_" +  f + "_nmads"] = nmads_abs
        tempdata.append(temp)
    to_add = ad.concat(tempdata)
    to_add = to_add.obs
    adata.obs = adata.obs.merge(to_add[[f + "_nmads", "abs_" +  f + "_nmads"]], left_index=True, right_index=True)
    # Plot nMAds on the UMAP
    sc.pl.umap(adata, color=f + "_nmads", frameon=True, save="_post_batch_post_batch_log10total_counts_nMAds.png")
    for l in lins:
        print(l)
        sc.pl.umap(adata[adata.obs.lineage == l], color=f + "_nmads", title=l, frameon=True, save="_" + l + "_post_batch_post_sweep_" + f + "_nmads" + ".png")
    # If we filter on abs(nMAds) < 2.5
    adata.obs[f + "_nmads_keep"] = np.abs(adata.obs[f + "_nmads"]) < 2.5
    adata.obs[f + "_nmads_keep"] = adata.obs[f + "_nmads_keep"].astype('category')
    for l in lins:
        print(l)
        sc.pl.umap(adata[adata.obs.lineage == l], color=f + "_nmads_keep", title=f + "_nmads_keep under 2.5", frameon=True, save="_" + l + "_post_batch_post_sweep_" + f + "_nmads_less_abs_2.5" + ".png")

# Plot distribution of these
ldir = data_name + "/" + status + "/" + category + "/figures/" + l
plt.figure(figsize=(8, 6))
l="lineage"
m=f + "_nmads"
fig,ax = plt.subplots(figsize=(8,6))
for r in lins:
    sns.distplot(adata.obs[adata.obs[l] == r][m], hist=False, rug=True, label=r)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Distribution of ' + m)
plt.xlabel(m + ' / cell')
plt.savefig(ldir + '/dist_' + m + '_per_cell_per_' + r + '_after_QC.png', bbox_inches='tight')
plt.clf()

# Raising the minimum nGene threshold to 300 may remove these strange immune-y cells
thresh = 350
adata.obs['gt_' + str(thresh) + '_nGenes'] = adata.obs['n_genes_by_counts'] > thresh
adata.obs['gt_' + str(thresh) + '_nGenes'] = adata.obs['gt_' + str(thresh) + '_nGenes'].astype('category')
for l in lins:
    print(l)
    sc.pl.umap(adata[adata.obs.lineage == l], color='gt_' + str(thresh) + '_nGenes', title=l + '_gt_' + str(thresh) + '_nGenes', frameon=True, save="_" + l + "_post_batch_post_sweep_nGene_thresh.png")
    

# Raising the minimum MT% cut off 
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
lins = np.unique(adata.obs.lineage)
for l in lins:
    sns.distplot(adata.obs[adata.obs.lineage == l].pct_counts_gene_group__mito_transcript, hist=False, rug=True, label=l)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Distribution of MT%')
plt.xlabel('MT% / cell')
#plt.axvline(x = 0.7, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/mt_perc_per_cell_per_lineage_after_QC.png', bbox_inches='tight')
plt.clf()

# How about a cut off of 1%
thresh = 1
adata.obs['gt_' + str(thresh) + '_MT'] = adata.obs['pct_counts_gene_group__mito_transcript'] > thresh
adata.obs['gt_' + str(thresh) + '_MT'] = adata.obs['gt_' + str(thresh) + '_MT'].astype('category')
sc.pl.umap(adata[adata.obs.pct_counts_gene_group__mito_transcript > thresh], color='category', title='MT > ' + str(thresh), frameon=True, save="_post_batch_post_sweep_MT_gt_" + str(thresh) + ".png")

for l in lins:
    print(l)
    sc.pl.umap(adata[adata.obs.lineage == l], color='gt_' + str(thresh) + '_MT', title=l + '_gt_' + str(thresh) + '_MT', frameon=True, save="_" + l + "_post_batch_post_sweep_MT_gt_" + str(thresh) + ".png")
    sc.pl.umap(adata[adata.obs.lineage == l], color='pct_counts_gene_group__mito_transcript', title=l, frameon=True, save="_" + l + "_post_batch_post_sweep_MT.png")
    

for l in lins:
    print(l)
    sc.pl.umap(adata[adata.obs.lineage == l], color='gt_' + str(thresh) + '_nGenes', title=l + '_gt_' + str(thresh) + '_nGenes', frameon=True, save="_" + l + "_post_batch_post_sweep_nGene_thresh.png")

####################################################################
########## Investigation ogf Azimuth/keras annotations #############
####################################################################
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

idx = adata.obs.lineage == "Immune"
use = adata.obs
use.columns = use.columns.str.strip()
use = use.rename(columns={"Azimuth:predicted.celltype.l2": "Azimuth_L2"})
concordance = pd.crosstab(use['label'][idx], use['Azimuth_L2'][idx])
#concordance.reset_index(inplace=True)
node_labels = concordance.index.astype(str)
link_source, link_target = np.meshgrid(range(29), range(29))
link_source, link_target = link_source.flatten(), link_target.flatten()
link_value = concordance.values.flatten()
num_rows = concordance.shape[0]
node_positions_left = np.arange(num_rows)
node_positions_right = np.arange(num_rows, 2 * num_rows)

# Creating Sankey plot
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=node_labels.tolist() * 2,  # Duplicate labels for both sides
        x=[0] * num_rows + [1] * num_rows,  # Assigning x=0 for left, x=1 for right
        y=np.concatenate([node_positions_left, node_positions_right])
    ),
    link=dict(
        source=np.concatenate([link_source, link_source + num_rows]),
        target=np.concatenate([link_target, link_target + num_rows]),
        value=link_value
    )
)])

# Adding title and labels
fig.update_layout(title_text='Two-Sided Concordance Matrix Sankey Plot', font_size=10)
fig.update_xaxes(title_text='Source')
fig.update_yaxes(title_text='Target')

# Save the plot as an image using plotly.io
image_bytes = pio.to_image(fig, format='png')
with open(figpath + "/two_sided_concordance_sankey_plot.png", "wb") as image_file:
    image_file.write(image_bytes)


####################################################################
#### Investigation of ngenes detected in each category/lineage #####
####################################################################
# Looks like there is a population of cells with relatively low ngenes expressed in Epithelium
# And these seem to overlap with those from the B Cell plasma with relatively high ngenes expressed in B cell plasma
# Plot the nMADs of these per category and lineage
feature="n_genes_by_counts"
lins = np.unique(adata.obs.lineage)
tempdata = []
for index,c in enumerate(lins):
    print(c)
    temp = adata[adata.obs.lineage == c]
    counts=temp.obs[feature]
    absolute_diff = np.abs(counts - np.median(counts))
    mad = np.median(absolute_diff)
    # Use direction aware filtering
    nmads = (counts - np.median(counts))/mad
    # Apply the function to change values in 'new_column' based on criteria
    temp.obs[feature + "_nmads"] = nmads
    # Also for absolute nMADs
    nmads_abs = absolute_diff/mad
    temp.obs["abs_" +  feature + "_nmads"] = nmads_abs
    tempdata.append(temp)

to_add = ad.concat(tempdata)
to_add = to_add.obs
adata.obs = adata.obs.merge(to_add[[feature + "_nmads", "abs_" +  feature + "_nmads"]], left_index=True, right_index=True)

# Plot this per lineage and overall
sc.pl.umap(adata, color=feature + "_nmads", title="nGenes_nMads", frameon=True, save="_post_batch_post_sweep_nGenes_nMads.png")
for l in lins:
    print(l)
    sc.pl.umap(adata[adata.obs.lineage == l], color=feature + "_nmads", title=l + " nGenes_nMads", frameon=True, save="_post_batch_post_sweep_" + l + "_nGenes_nMads.png")

# Plot as a distribution (direction aware)
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
lins = np.unique(adata.obs.lineage)
for l in lins:
    sns.distplot(adata.obs[adata.obs.lineage == l].n_genes_by_counts_nmads, hist=False, rug=True, label=l)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Distribution of ngenes expressed (nMADs)')
plt.xlabel('nMAD Genes expressed / cell')
#plt.axvline(x = 0.7, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/ngenes_expressed_per_cell_per_lineage_after_QC_nMads.png', bbox_inches='tight')
plt.clf()

# If we were to set a threshold of -2.5 and +2.5 nMAds? (so an absolute of 2.5, like the MT%)
min, max = -2.5,2.5
adata.obs['ngene_nMads_keep'] = (adata.obs.n_genes_by_counts_nmads > min) & (adata.obs.n_genes_by_counts_nmads < max)
adata.obs['ngene_nMads_keep'] = adata.obs['ngene_nMads_keep'].astype('category')
sc.pl.umap(adata, color='ngene_nMads_keep', title="nGenes_nMads_keep", frameon=True, save="_post_batch_post_sweep_nGenes_nMads_keep.png")
for l in lins:
    print(l)
    sc.pl.umap(adata[adata.obs.lineage == l], color="ngene_nMads_keep", title=l + " ngene_nMads_keep", frameon=True, save="_post_batch_post_sweep_" + l + "_nGenes_nMads_keep.png")

# Check the cells remaining if we apply this threshold
adata_strict = adata[adata.obs.abs_n_genes_by_counts_nmads < 2.5]
for c in cats:
    print(c)
    before = adata.obs[adata.obs.category == c].shape[0]
    after = adata_strict.obs[adata_strict.obs.category == c].shape[0]
    loss = before - after
    loss_perc = 100*(loss/before)
    print("From {}, we lose {} cells ({:.2f}%)".format(c, loss, loss_perc))
    sc.pl.umap(adata_strict[adata_strict.obs.category == c], color='label', title=c + " ngene_nMads_keep (abs < 2.5)", frameon=True, save="_post_batch_post_sweep_" + c + "_nGenes_nMads_keep.png")

# Check enterocyte/B Cell markers in these cells
gene_ens = ["ENSG00000119888", "ENSG00000039068", "ENSG00000171345", "ENSG00000170476", "ENSG00000132465"]
gene_symb = ["EPCAM", "CDH1", "KRT19", "MZB1", "JCHAIN"]
for index, g in enumerate(gene_symb):
    sc.pl.umap(adata[adata.obs.category == "B Cell plasma"], color=gene_ens[index], frameon=False , title=g, save="_" + g + "_BCP.png")
    sc.pl.umap(adata[adata.obs.category == "Enterocyte"], color=gene_ens[index], frameon=False , title=g, save="_" + g + "_Enterocytes.png")

    
#####################################################
######## Investigation of low MT% Immune ############
#####################################################
# After using 2.5 absolute MAD as a threshold, this removes the low MT% epithelium (below)
# This does however retain a population of Myeloid, B Cell and T cells with very low MT%
# These seem to be mixing on the basis of this
# Cluster these and have a look at their markers
test_immune = adata[adata.obs.category.isin(["Myeloid", "T Cell", "B Cell"])]

# PErform some low-level (low res) clustering of these cells (takes ~ 2hours)
sc.tl.leiden(test_immune, resolution=0.2, neighbors_key ="scVI_nn")

# Plot
sc.pl.umap(test_immune, color="leiden", frameon=True, save="_post_batch_post_batch_coarse_immune_leiden.png",legend_loc='on data')

# Check is number 7
sc.pl.umap(test_immune[test_immune.obs.leiden == "7"], color="leiden", frameon=True, save="_post_batch_post_batch_coarse_immune_leiden_cl7.png",legend_loc='on data')

# Extract cluster 7 and subcluster
cl7 = test_immune[test_immune.obs.leiden == "7"]
sc.tl.leiden(cl7, resolution=0.3, neighbors_key ="scVI_nn")
sc.pl.umap(cl7, color="leiden", frameon=True, save="_post_batch_post_batch_coarse_immune_leiden_cl7_subclustered.png",legend_loc='on data')

# Clusters 1 and 2 seem to match the strange cluster.
# Add this onto the total adata and find markers
strange_ids = cl7[cl7.obs.leiden.isin(["1","2"])].obs.index
adata.obs['low_MT_strange_cluster'] = adata.obs.index.isin(strange_ids)
adata.obs['low_MT_strange_cluster'] = adata.obs['low_MT_strange_cluster'].astype('category')
sc.pl.umap(adata, color="low_MT_strange_cluster", frameon=True, save="_post_batch_post_batch_coarse_strange_low_MT_cluster.png")
sc.tl.rank_genes_groups(adata, groupby="low_MT_strange_cluster", method='t-test', groups = ["True"], use_raw=True)
sc.tl.rank_genes_groups(adata, 'low_MT_strange_cluster', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save="_markers_lowMT_strange_cluster.png")

# Get results out
results = adata.uns['rank_genes_groups']
# Create a DataFrame for the results of the specified group
True_markers = pd.DataFrame({
    'names': results['names']['True'],
    'logFC': results['logfoldchanges']['True'],
    'pvals': results['pvals']['True'],
    'pvals_adj': results['pvals_adj']['True'],
})
True_markers = True_markers[True_markers.pvals_adj < 0.05]

# Add names
ens_sym = adata.var[['gene_symbols']]
ens_sym = ens_sym.reset_index()
ens_sym = ens_sym.rename(columns={'index': 'names'})
True_markers = True_markers.merge(ens_sym, on="names")
# Remove RP and save
True_markers = True_markers[~(True_markers.gene_symbols.str.contains("^RP"))]
True_markers.to_csv(figpath + "/markers_strange_low_MT_Celltype.csv")

# Plot markers
sc.pl.rank_genes_groups_violin(adata, groups='True', n_genes=8, save="_strange_low_MT_cluster_markers.png")


# Perform some enrichment
from gprofiler import GProfiler
gene_list = True_markers[True_markers.logFC > 2].gene_symbols.values.tolist()
gp = GProfiler(return_dataframe=True)  # Set return_dataframe to True to get results as a DataFrame
results = gp.profile(organism='hsapiens', query=gene_list, sources=['GO:BP'], user_threshold=0.05)
results.to_csv(figpath + "/GO_BP_abslogFC_gt_2_strange_low_MT_cluster.csv")



#####################################################
######## Investigation of low MT% epithelium ########
#####################################################
# If we subset using a mixture model (see below) or using a direction aware MAD cut off of MT%, there remains an epithelial population (enterocytes and stem cells) with almost NO
# MT%. Find this very suspect and potentially an indicator of poorly sequenced cells, or cells without mitochondria (maybe burst?)
# Subset for the epithelial populations
epi = adata[adata.obs.lineage == "Epithelial"]

# PErform some low-level (low res) clustering of these cells (takes ~ 2hours)
sc.tl.leiden(epi, resolution=0.2, neighbors_key ="scVI_nn")

# Plot
sc.pl.umap(epi, color="leiden", frameon=True, save="_post_batch_post_epithelial_leiden_0.2.png")

# Save annot
epi.obs[['leiden']].to_csv(tabdir + "/epoithelial_only_leiden_0.2.csv")

# Find markers of the rogue cell-type
epi.uns['log1p'] = None 
sc.tl.rank_genes_groups(epi, groupby="cell_type_annotation_column", method='t-test', groups = ["4"], reference="rest", use_raw=True)


#####################################################

# Plot RP%
sc.pl.umap(adata, color="pct_counts_gene_group__ribo_protein", frameon=True, save="_post_batch_post_sweep_rpperc.png")
# Calculate IG% and plot
adata.var[adata.var.gene_symbols.str.contains("IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]")]
totsum = adata.layers['counts'].sum(axis=1)
is_ig = adata.var.gene_symbols.str.contains("IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]")
temp_counts = adata.layers['counts']
igcounts = temp_counts[:,is_ig.values]
igsum = igcounts.sum(axis=1)
ig_perc = np.array(igsum)/np.array(totsum)
adata.obs['ig_perc'] = ig_perc
sc.pl.umap(adata, color="ig_perc", frameon=True, save="_post_batch_post_sweep_igperc.png")


# Plot within category, coloured by keras confidence, label, MTperc, for each embedding
latents = ["X_pca", "X_scANVI", "X_scVI_default", "X_scVI", 'X_pca_harmony']
cats = np.unique(adata.obs.category)
for l in latents:
    print(l)
    adata.obsm['X_umap'] = adata.obsm['UMAP_' + l]
    for c in cats:
        print(c)
        temp = adata[adata.obs.category == c]
        sc.pl.umap(temp, color="Keras:predicted_celltype_probability", title=c, frameon=False, save="_" + l + "_" + c + "_post_batch_post_sweep_keras_conf.png")
        sc.pl.umap(temp, color="label", title=c, frameon=False , save="_" + l + "_" + c + "_post_batch_post_sweep_label.png")
        sc.pl.umap(temp, color="pct_counts_gene_group__mito_transcript", title=c, frameon=False , save="_" + l + "_" + c + "_post_batch_post_sweep_mt_perc.png")
        sc.pl.umap(temp, color="n_genes_by_counts", frameon=True, save="_" + l + "_" + c + "_post_batch_post_ngenes.png")
        sc.pl.umap(temp, color="log10ngenes", frameon=True, save="_" + l + "_" + c + "_post_batch_log10ngenes.png")
        sc.pl.umap(temp, color="pct_counts_gene_group__ribo_protein", frameon=True, save="_" + l + "_" + c + "_post_batch_rp_perc.png")

# Has created lots of files. Sort these out
import shutil
for l in latents:
    print(l)
    lfigdir=figpath + "/" + l
    print(lfigdir)
    if not os.path.exists(lfigdir):
        os.mkdir(lfigdir)
    files = os.listdir(figpath)
    for file in files:
        if l in file and ".png" in file:
            print(file)
            src = os.path.join(figpath, file)
            dest = os.path.join(lfigdir, file)
            shutil.move(src, dest)
            print("Moved")

# Correct "X_pca_harmony" and "X_scVI_default"
for file in os.listdir("rectum/healthy/All_not_gt_subset/figures/X_pca"):
    if "harmony" in file:
        src = os.path.join("rectum/healthy/All_not_gt_subset/figures/X_pca", file)
        dest = os.path.join("rectum/healthy/All_not_gt_subset/figures/X_pca_harmony", file)
        shutil.move(src, dest)
        print("Moved")

for file in os.listdir("rectum/healthy/All_not_gt_subset/figures/X_scVI"):
    if "default" in file:
        src = os.path.join("rectum/healthy/All_not_gt_subset/figures/X_scVI", file)
        dest = os.path.join("rectum/healthy/All_not_gt_subset/figures/X_scVI_default", file)
        shutil.move(src, dest)
        print("Moved")
        

#####################################################
######## Exploration of B cell plasma cells #########
#####################################################


# have a look at the MT% per B cell plasma cell type
temp = adata[adata.obs.category == "B Cell plasma"]
sns.set_palette('tab10')
labs=np.unique(adata.obs.label)
labs = [item for item in labs if "plasma IgA" in item]
mt_per_cat = pd.DataFrame({"label": labs, "95L": 0, "mean": 0, "95U": 0})
for index, c in enumerate(labs):
    data = temp.obs[temp.obs.label == c].pct_counts_gene_group__mito_transcript
    lims = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    mean = sum(data)/len(data)
    mt_per_cat.loc[mt_per_cat.label == c, "95L"] = lims[0]
    mt_per_cat.loc[mt_per_cat.label == c, "mean"] = mean
    mt_per_cat.loc[mt_per_cat.label == c, "95U"] = lims[1]
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(data, hist=False, rug=True, label=c)

plt.legend()
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(figpath + '/mt_perc_50pct_per_Bcellplasma_label_postQC.png', bbox_inches='tight')
plt.clf()

# if we plot the B cell plasma cells with high MT%
temp.obs['B_cell_plasma_low_MT'] = (temp.obs['label'] == "B cell plasma IgA (3)") & (temp.obs['pct_counts_gene_group__mito_transcript'] < 10)
temp.obs['B_cell_plasma_low_MT'] = temp.obs['B_cell_plasma_low_MT'].astype('category')
temp.obs['is_B_cell_plasma3'] = (temp.obs['label'] == "B cell plasma IgA (3)")
temp.obs['is_B_cell_plasma3'] = temp.obs['is_B_cell_plasma3'].astype('category')
sc.pl.umap(temp, color="is_B_cell_plasma3", frameon=False , save="_" + c + "_post_batch_post_sweep_is_B_cell_plasma.png")
sc.pl.umap(temp, color="B_cell_plasma_low_MT", frameon=False , save="_" + c + "_post_batch_post_sweep_B_cell_plasma_low_MT.png")

# Have a look at the next-best guess for these cells - perhaps their confidence is close
# Plot keras first-second confidence (y) versus MT% (x), coloured by the category of their second annotation
dir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/2023_08_07/rectum/results/celltype/keras_celltype"
data = []
samples = np.unique(adata.obs.convoluted_samplename)
for i in range(0,len(samples)):
    print(samples[i])
    temp = ad.read_h5ad(dir + "/" + samples[i] + "/" + samples[i] + "___cellbender_fpr0pt1-scrublet-ti_freeze003_prediction-predictions.h5ad")
    want = temp.obs.columns.str.contains("probability__")
    temp_probs = temp.obs.loc[:,want]
    data.append(temp_probs)
    del temp, temp_probs

keras = pd.concat(data, ignore_index=False)
keras = keras[keras.index.isin(adata.obs.index)]

# Extract the first and second-best annotations
def find_most_confident(row):
    sorted_row = row.sort_values(ascending=False)
    most_confident = sorted_row.index[0]
    second_most_confident = sorted_row.index[1]
    confidence = sorted_row[0]
    second_confidence = sorted_row[1]
    return most_confident, confidence, second_most_confident, second_confidence

keras_top2 = keras.apply(find_most_confident, axis=1, result_type='expand')
keras_top2.columns = ['Most_Confident_Cell', 'Confidence1', 'Second_Most_Confident_Cell', 'Confidence2']

# Add the category of these
keras_top2['Most_Confident_Cell'] = keras_top2['Most_Confident_Cell'].apply(lambda x: x.replace('probability__', ''))
keras_top2['Second_Most_Confident_Cell'] = keras_top2['Second_Most_Confident_Cell'].apply(lambda x: x.replace('probability__', ''))
add_category = adata.obs[["label__machine", "category"]]
add_category = add_category.reset_index()
add_category = add_category[["label__machine", "category"]]
add_category = add_category.drop_duplicates()
add_category.rename(columns={'label__machine': 'Most_Confident_Cell', 'category': 'Most_Confident_Cell_category'}, inplace=True)
keras_top2 = keras_top2.reset_index()
keras_top2 = keras_top2.merge(add_category, on="Most_Confident_Cell")
add_category.rename(columns={'Most_Confident_Cell': 'Second_Most_Confident_Cell', 'Most_Confident_Cell_category': 'Second_Most_Confident_Cell_category'}, inplace=True)
keras_top2 = keras_top2.merge(add_category, on="Second_Most_Confident_Cell")
keras_top2['Same_category'] = keras_top2['Most_Confident_Cell_category'] == keras_top2['Second_Most_Confident_Cell_category']
keras_top2['Same_category'] = keras_top2['Same_category'].astype('category')
keras_top2['First_confidence-Second_confidence'] = keras_top2['Confidence1'] - keras_top2['Confidence2']
keras_top2['First_over_Second_confidence'] = np.log10(keras_top2['Confidence1'] / keras_top2['Confidence2'])

# Add lineage
cats = np.unique(adata.obs.category)
lin_df = pd.DataFrame({'category': cats, 'lineage': ""}) 
lin_df['lineage'] = np.where(lin_df['category'].isin(['B Cell', 'B Cell plasma', 'T Cell', 'Myeloid']), 'Immune', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category'].isin(['Stem cells', 'Secretory', 'Enterocyte']), 'Epithelial', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category']== 'Mesenchymal', 'Mesenchymal', lin_df['lineage'])
lin_df.rename(columns={'category': 'Most_Confident_Cell_category', 'lineage': 'Most_Confident_Cell_lineage'}, inplace=True)
keras_top2 = keras_top2.merge(lin_df, on="Most_Confident_Cell_category")
lin_df.rename(columns={'Most_Confident_Cell_category': 'Second_Most_Confident_Cell_category', 'Most_Confident_Cell_lineage': 'Second_Most_Confident_Cell_lineage'}, inplace=True)
keras_top2 = keras_top2.merge(lin_df, on="Second_Most_Confident_Cell_category")
keras_top2.set_index('index', inplace=True)
keras_top2['Same_lineage'] = keras_top2['Most_Confident_Cell_lineage'] == keras_top2['Second_Most_Confident_Cell_lineage']
keras_top2['Same_lineage'] = keras_top2['Same_lineage'].astype('category')


# Have a look at the difference of IgA (3) B cell plasma cells in confident set and the next best guess
keras_top2_qc_bcp_3 = keras_top2[keras_top2['Most_Confident_Cell'] == 'B_cell_plasma_IgA_CD38plusplusplus']
keras_top2_qc_bcp_3.set_index('index', inplace=True)
# Make a plot comparing the MT% and keras confidence for this cell type
test = adata.obs[adata.obs.label == "B cell plasma IgA (3)"]
test = test.merge(keras_top2_qc_bcp_3, left_index=True, right_index=True)

# Plot
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for c in cats:
    testfilt = test[test['Second_Most_Confident_Cell_category'] == c]
    plt.scatter(testfilt['pct_counts_gene_group__mito_transcript'], testfilt['First_over_Second_confidence'], alpha=1, s=20)

plt.title('B cell plasma IgA (3) cells - coloured by category of 2nd annotation')
plt.legend(cats, bbox_to_anchor=(1.0, 1.0))
plt.xlabel('MT%')
plt.ylabel('log10(Confidence of top keras annotation / second)')
plt.savefig(figpath + '/mt_perc_keras_conf_BCP_3.png', bbox_inches='tight')
plt.clf()

# Plot distribution of second annotation category with respect to MT%
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for c in cats:
    testfilt = test[test['Second_Most_Confident_Cell_category'] == c]
    sns.distplot(testfilt['pct_counts_gene_group__mito_transcript'], hist=False, rug=True, label=c)

plt.title('B cell plasma IgA (3) cells - coloured by category of 2nd annotation')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(figpath + '/mt_perc_keras_conf_BCP_3_dist.png', bbox_inches='tight')
plt.clf()

# Summarise the lineage confidence of these cells and plot
keras = keras.T
keras = keras.reset_index()
keras = keras.rename(columns = {'index': 'label__machine'})
keras['label__machine'] = keras['label__machine'].apply(lambda x: x.replace('probability__', ''))
add_category.rename(columns = {'Second_Most_Confident_Cell': 'label__machine', 'Second_Most_Confident_Cell_category': 'category'}, inplace = True)
keras = keras.merge(add_category, on='label__machine')
lin_df.rename(columns = {'Second_Most_Confident_Cell_category' : 'category', 'Second_Most_Confident_Cell_lineage': 'lineage'}, inplace=True)
keras = keras.merge(lin_df, on='category')
keras.set_index('label__machine', inplace=True)
keras_per_lin = keras.groupby('lineage').sum()
keras_per_lin.drop('category', axis=1, inplace=True)
keras_per_lin = keras_per_lin.T
# How many cells have a majority per-lineage confidence that is large
cutoff=float(0.8)
max_vals = keras_per_lin[['Epithelial', 'Immune', 'Mesenchymal']].max(axis=1)
keras_per_lin['lin_max_gt_' + str(cutoff)] = max_vals > cutoff

# Plot the distribution of these
keras_per_lin['Max_Category'] = keras_per_lin[['Epithelial', 'Immune', 'Mesenchymal']].apply(lambda x: x.idxmax(), axis=1)
lins = np.unique(keras_per_lin.Max_Category)
# Plot distribution of second annotation category with respect to MT%
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    testfilt = keras_per_lin[keras_per_lin['Max_Category'] == l]
    sns.distplot(testfilt[l], hist=False, rug=True, label=l)

plt.title('Distribution of Keras confidence per lineage - All cells')
plt.legend(bbox_to_anchor=(1.0, 1.0))
#plt.axvline(float(cutoff), color='red', linestyle='dotted', label='Vertical Line')
ax.set(xlim=(0, 1))
plt.xlabel('Combined keras confidence per lineage')
plt.savefig(figpath + '/keras_conf_per_lineage_dist.png', bbox_inches='tight')
plt.clf()

# Plot distribution of B cell plasma cells
bcp_per_lin = keras_per_lin[keras_per_lin.index.isin(keras_top2_qc_bcp_3.index)]
test = test.merge(bcp_per_lin, left_index=True, right_index=True)
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    testfilt = bcp_per_lin[bcp_per_lin['Max_Category'] == l]
    sns.distplot(testfilt[l],  hist=False, rug=True, label=l)

plt.title('B cell plasma IgA (3) cells lineage confidence')
plt.legend(bbox_to_anchor=(1.0, 1.0))
ax.set(xlim=(0, 1))
plt.xlabel('Summed probability of major lineage')
plt.savefig(figpath + '/keras_conf_BCP_3_lineage.png', bbox_inches='tight')
plt.clf()

# Plot this data on a umap
adata.obs = adata.obs.merge(keras_per_lin, left_index=True, right_index=True)
for l in lins:
    sc.pl.umap(adata, color=l, frameon=False , save="_post_batch_post_sweep_" + l + "_probability.png")

# Plot all (and B plasma cells only) by their next-best guess category and lineage
adata.obs = adata.obs.merge(keras_top2, left_index=True, right_index=True)
sc.pl.umap(adata, color="Second_Most_Confident_Cell_category", frameon=False , save="_post_batch_post_Second_Most_Confident_Cell_category.png")
sc.pl.umap(adata, color="Second_Most_Confident_Cell_lineage", frameon=False , save="_post_batch_post_Second_Most_Confident_Cell_lineage.png")
sc.pl.umap(adata, color="Most_Confident_Cell_category", frameon=False , save="_post_batch_post_Most_Confident_Cell_category.png")
sc.pl.umap(adata, color="Most_Confident_Cell_lineage", frameon=False , save="_post_batch_post_Most_Confident_Cell_lineage.png")


# Make a plot for the loss of different cell types/cut off
sequence = [i * 0.1 for i in range(11)]
lin_cuts = pd.DataFrame({"cutoff": sequence, "nEpithelial":0,  "percEpithelial": 0, "nImmune": 0, "percImmune": 0, "nMesenchymal": 0, "percMesenchymal": 0})
for l in lins:
    temp = adata.obs[adata.obs.Max_Category == l]
    norig = temp.shape[0]
    for c in lin_cuts['cutoff'].values:
        nremain = temp[temp[l] < c].shape[0]
        lin_cuts.loc[lin_cuts.cutoff == c,'n' + l] = norig - nremain
        lin_cuts.loc[lin_cuts.cutoff == c,'perc' + l] = 100-100*(nremain/norig)

plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    plt.plot(lin_cuts.cutoff, lin_cuts['perc' + l])

plt.legend(lins, bbox_to_anchor=(1.0, 1.0))
plt.title('Loss of cells based on lineage confidence')
plt.xlabel('Probability cut off')
plt.ylabel('Retained (%)')
plt.axvline(x = 0.7, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/lin_perc_cutoffs.png', bbox_inches='tight')
plt.clf()

# Do Plot frequencies
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    plt.plot(lin_cuts.cutoff, lin_cuts['n' + l])

plt.legend(lins, bbox_to_anchor=(1.0, 1.0))
plt.title('Loss of cells based on lineage confidence')
plt.xlabel('Probability cut off')
plt.ylabel('Retained (Absolute number)')
plt.axvline(x = 0.7, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/lin_perc_cutoffs_freqs.png', bbox_inches='tight')
plt.clf()


# Plot a UMAP without these cells, coloured by cell
lin_conf_cutoff = 0.7
max_vals = adata.obs[['Epithelial', 'Immune', 'Mesenchymal']].max(axis=1)
adata.obs['lin_max_gt_' + str(lin_conf_cutoff)] = max_vals > cutoff
adata_high_filt = adata[adata.obs['lin_max_gt_' + str(lin_conf_cutoff)] == True]
sc.pl.umap(adata_high_filt, color='category', frameon=False , save="_post_batch_post_sweep_post_lin_filt_label.png")

# Within category
for c in cats:
    temp_high = adata_high_filt[adata_high_filt.obs.category == c]
    sc.pl.umap(temp_high, color='label', frameon=False , title=c, save="_" + c + "_post_batch_post_sweep_post_lin_filt_label.png")

# This doesn't seem to have done too much in terms of addressing these cells
# So will have a look at using a higher cut off as a last ditch atempt for the lineage probability approach
lin_conf_cutoff = 0.9
adata.obs['lin_max_gt_' + str(lin_conf_cutoff)] = max_vals > lin_conf_cutoff
adata_high_filt = adata[adata.obs['lin_max_gt_' + str(lin_conf_cutoff)] == True]
sc.pl.umap(adata_high_filt, color='category', frameon=False , save="_post_batch_post_sweep_post_lin_filt" + str(lin_conf_cutoff) + "_label.png")
for c in cats:
    temp_high = adata_high_filt[adata_high_filt.obs.category == c]
    sc.pl.umap(temp_high, color='label', frameon=False , title=c, save="_" + c + "_post_batch_post_sweep_post_lin_filt_" + str(lin_conf_cutoff) + "_label.png")

#### Scoring cells based on first vs second conf
def calculate_confidence_score(row):
    if row['Confidence1'] >= 0.9:
        return 'Very_high'
    elif 0.5 <= row['Confidence1'] and row['Same_lineage'] == True:
        return 'High'
    elif 0.5 <= row['Confidence1'] and row['Same_lineage'] == False:
        return 'Medium'
    elif row['Confidence1'] < 0.5 and row['Same_lineage'] == False:
        return 'Low'
    else:
        return 'Very_low'

# Apply the function to create the 'confidence_score' column
adata.obs['confidence_score'] = adata.obs.apply(calculate_confidence_score, axis=1)
adata.obs['confidence_score'] = pd.Categorical(adata.obs['confidence_score'], categories=["Very_high", "High", "Medium", "Low", "Very_low"], ordered=True)
sc.pl.umap(adata, color='confidence_score', frameon=False , save="_post_batch_post_sweep_confidence_score.png")

# Plot this per category
for c in cats:
    tempadata = adata[adata.obs.category == c]
    sc.pl.umap(tempadata, color='confidence_score', frameon=False , title=c, save="_" + c + "_post_batch_post_sweep_post_confidence_score.png")

notlow = adata[adata.obs.confidence_score.isin(["Very_high", "High", "Medium"])]
sc.pl.umap(notlow, color='confidence_score', frameon=False , save="_post_batch_post_sweep_confidence_score_notlow.png")
for c in cats:
    tempadata = notlow[notlow.obs.category == c]

# Have a look at performing some lineage based cut off on the basis of MT%
def relative_mt_per_lineage(row):
    if row['lineage'] == lineage and row['pct_counts_gene_group__mito_transcript'] > relative_threshold:
        return 'F'  # Modify the value
    else:
        return row['keep']  # Keep the value unchanged

# Count the ndeviations (direction aware)
tempdata = []
features = ["pct_counts_gene_group__mito_transcript", "pct_counts_gene_group__ribo_protein"]
feature_names = ["mt_perc", "ribo_perc"]
for index,c in enumerate(lins):
    print(c)
    temp = adata[adata.obs.lineage == c]
    for id2, f in enumerate(features):
        feature = temp.obs[features[id2]]
        absolute_diff = np.abs(feature - np.median(feature))
        mad = np.median(absolute_diff)
        nmads = (feature - np.median(feature))/mad
        temp.obs[feature_names[id2] + "_nmads"] = nmads
    tempdata.append(temp)

to_add = ad.concat(tempdata)
to_add = to_add.obs
adata.obs = adata.obs.merge(to_add[[feature_names[0] + "_nmads", feature_names[1] + "_nmads"]], left_index=True, right_index=True)


# Plotting RP% per lineage
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    temp = adata[adata.obs.lineage == l]
    sns.distplot(temp.obs['pct_counts_gene_group__ribo_protein'], hist=False, rug=True, label=l)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('RP% per lineage')
plt.xlabel('RP%')
ax.set(xlim=(0, 50))
plt.savefig(figpath + '/lin_rp_perc.png', bbox_inches='tight')
plt.clf()

# Plot the distribution of MADs for each lineage
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    temp = adata[adata.obs.lineage == l]
    sns.distplot(temp.obs['ribo_perc_nmads'], hist=False, rug=True, label=l)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Median absolute deviations of MT% per lineage')
plt.xlabel('Median absolute deviation')
#plt.axvline(x = 0.7, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/lin_rp_per_mads.png', bbox_inches='tight')
plt.clf()

# Plotting MT% per lineage
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    temp = adata[adata.obs.lineage == l]
    sns.distplot(temp.obs['pct_counts_gene_group__mito_transcript'], hist=False, rug=True, label=l)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('MT% per lineage')
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(figpath + '/lin_mt_perc.png', bbox_inches='tight')
plt.clf()

# Plot the distribution of MADs for each lineage
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for l in lins:
    temp = adata[adata.obs.lineage == l]
    sns.distplot(temp.obs['mt_perc_nmads'], hist=False, rug=True, label=l)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Median absolute deviations of MT% per lineage')
plt.xlabel('Median absolute deviation')
#plt.axvline(x = 0.7, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/lin_mt_per_mads.png', bbox_inches='tight')
plt.clf()

# How would a cut off of 2.5 affect cell proportions
relative_threshold = 2.5
for f in feature_names:
    adata.obs['abs_' + f + '_nmads'] = np.abs(adata.obs[f + '_nmads'])
    adata.obs['abs_' + f + '_nmads'] = np.abs(adata.obs['abs_' + f + '_nmads'])
    adata.obs['abs_' + f + '_nmads' + '_gt' + str(relative_threshold)] = adata.obs['abs_' + f + '_nmads'] > relative_threshold
    adata.obs['abs_' + f + '_nmads' + '_gt' + str(relative_threshold)] = adata.obs['abs_' + f + '_nmads' + '_gt' + str(relative_threshold)].astype('category')
    sc.pl.umap(adata, color='abs_' + f + '_nmads' + '_gt' + str(relative_threshold), frameon=False , title='abs_' + f + '_nmads' + '_gt' + str(relative_threshold), save="_post_batch_post_sweep_post_abs_" + f + '_nmads' + '_gt' + str(relative_threshold) +".png")
    for c in cats:
        tempadata = adata[adata.obs.category == c]
        sc.pl.umap(tempadata, color='abs_' + f + '_nmads' + '_gt' + str(relative_threshold), frameon=False , title=c + '_' + 'abs_' + f + '_nmads' + '_gt' + str(relative_threshold), save="_" + c + "_post_batch_post_sweep_post_abs_" + f + '_nmads' + '_gt' + str(relative_threshold) +".png")

# Have a look per category
for c in cats:
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color='mt_perc_nmads', frameon=False , title=c, save="_" + c + "_post_batch_post_sweep_mt_perc_nmads.png")

# Plot MT% 
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for c in cats:
    temp = adata[adata.obs.category == c]
    sns.distplot(temp.obs['pct_counts_gene_group__mito_transcript'], hist=False, rug=True, label=c)

plt.legend()
plt.title('MT% post QC')
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(figpath + '/postqc_mtperc_per_cat.png', bbox_inches='tight')
plt.clf()

# Fit a gaussian mixture model to this data
# What if we subset for cels with MT% > 0.5
not_low = adata[adata.obs.pct_counts_gene_group__mito_transcript > 0.5]
sc.pl.umap(not_low, color='category', frameon=False , title="MT% > 0.5", save="_post_batch_post_sweep_mt_gt_0.5.png")

from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sympy import symbols, Eq, solve
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
from scipy.optimize import brentq
for c in cats:
    # Extract data
    data = adata.obs[adata.obs.category == c].pct_counts_gene_group__mito_transcript.values.reshape(-1, 1)
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    # 1. Fit a GMM with multiple initializations
    gmm = GaussianMixture(n_components=2, n_init=10).fit(scaled_data.reshape(-1, 1))
    # Convert means and variances back to original space
    original_means = scaler.inverse_transform(gmm.means_).flatten()
    original_variances = gmm.covariances_.flatten() * (scaler.scale_**2)
    # 2. Calculate fold change
    fold_change = max(original_means) / min(original_means)
    log_fold_change = np.log2(fold_change)
    print("Fold Change:", fold_change)
    print("Log2 Fold Change:", log_fold_change)
    # 3. Plot the data
    x = np.linspace(min(data), max(data), 1000)
    scaled_x = scaler.transform(x.reshape(-1, 1)).flatten()
    weights = gmm.weights_
    pdf1 = weights[0] * (1/(np.sqrt(2*np.pi*original_variances[0]))) * np.exp(-0.5 * ((x-original_means[0])**2)/original_variances[0])
    pdf2 = weights[1] * (1/(np.sqrt(2*np.pi*original_variances[1]))) * np.exp(-0.5 * ((x-original_means[1])**2)/original_variances[1])
    total_pdf = pdf1 + pdf2
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    plt.plot(x, pdf1, label='Distribution 1')
    plt.plot(x, pdf2, label='Distribution 2')
    plt.plot(x, total_pdf, 'k--', label='Mixture Model')
    plt.legend()
    plt.title(f"{c}: MT percentage mixture models, FC={fold_change:.4f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(figpath + '/mixture_model_MT_perc_postqc_' + c + '.png', bbox_inches='tight')
    plt.clf()
    # If FC > 10:, find the intersection
    if fold_change > 10:
        def difference(z):
            pdf1_val = weights[0] * (1/(np.sqrt(2*np.pi*original_variances[0]))) * np.exp(-0.5 * ((z-original_means[0])**2)/original_variances[0])
            pdf2_val = weights[1] * (1/(np.sqrt(2*np.pi*original_variances[1]))) * np.exp(-0.5 * ((z-original_means[1])**2)/original_variances[1])
            return pdf1_val - pdf2_val
            
            # Define a function that restricts the search space to values less than 10
        lower_bound = 0
        upper_bound = 10
        try:
            intersection = brentq(difference, lower_bound, upper_bound)
            print("Intersection:", intersection)
        except ValueError:
            print("No intersection found in the given range.")

# Have a look at the B Cell plasma cells above and below this threshold
bcp_only = adata[adata.obs.category == "B Cell plasma"]
bcp_only.obs['BCP_IGA_gt_MM_int'] = bcp_only.obs['pct_counts_gene_group__mito_transcript'] > intersection
bcp_only.obs['BCP_IGA_gt_MM_int'] = bcp_only.obs['pct_counts_gene_group__mito_transcript'] > intersection
bcp_only.obs['BCP_IGA_gt_MM_int'] = bcp_only.obs['BCP_IGA_gt_MM_int'].astype('category')
sc.pl.umap(bcp_only, color='BCP_IGA_gt_MM_int', frameon=False , title="Above or below intersection of MM for MT%", save="_BCP_above_MT_mm.png")
np.unique(bcp_only.obs['BCP_IGA_gt_MM_int'], return_counts=True)

# Plot marker genes
gene_ens = ["ENSG00000119888", "ENSG00000039068", "ENSG00000171345", "ENSG00000170476", "ENSG00000132465"]
gene_symb = ["EPCAM", "CDH1", "KRT19", "MZB1", "JCHAIN"]
for index, g in enumerate(gene_symb):
    sc.pl.umap(adata[adata.obs.category == "B Cell plasma"], color=gene_ens[index], frameon=False , title=g, save="_" + g + "_BCP.png")
    sc.pl.umap(adata[adata.obs.category == "Enterocyte"], color=gene_ens[index], frameon=False , title=g, save="_" + g + "_Enterocytes.png")
    sc.pl.umap(adata[adata.obs.category == "Stem cells"], color=gene_ens[index], frameon=False , title=g, save="_" + g + "_Stem.png")

# What if we have a look at this on the basis of RP%

from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sympy import symbols, Eq, solve
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
from scipy.optimize import brentq
for c in cats:
    # Extract data
    data = adata.obs[adata.obs.category == c].pct_counts_gene_group__ribo_protein.values.reshape(-1, 1)
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    # 1. Fit a GMM with multiple initializations
    gmm = GaussianMixture(n_components=2, n_init=10).fit(scaled_data.reshape(-1, 1))
    # Convert means and variances back to original space
    original_means = scaler.inverse_transform(gmm.means_).flatten()
    original_variances = gmm.covariances_.flatten() * (scaler.scale_**2)
    # 2. Calculate fold change
    fold_change = max(original_means) / min(original_means)
    log_fold_change = np.log2(fold_change)
    print("Fold Change:", fold_change)
    print("Log2 Fold Change:", log_fold_change)
    # 3. Plot the data
    x = np.linspace(min(data), max(data), 1000)
    scaled_x = scaler.transform(x.reshape(-1, 1)).flatten()
    weights = gmm.weights_
    pdf1 = weights[0] * (1/(np.sqrt(2*np.pi*original_variances[0]))) * np.exp(-0.5 * ((x-original_means[0])**2)/original_variances[0])
    pdf2 = weights[1] * (1/(np.sqrt(2*np.pi*original_variances[1]))) * np.exp(-0.5 * ((x-original_means[1])**2)/original_variances[1])
    total_pdf = pdf1 + pdf2
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    plt.plot(x, pdf1, label='Distribution 1')
    plt.plot(x, pdf2, label='Distribution 2')
    plt.plot(x, total_pdf, 'k--', label='Mixture Model')
    plt.legend()
    plt.title(f"{c}: RP percentage mixture models, FC={fold_change:.4f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(figpath + '/mixture_model_RP_perc_postqc_' + c + '.png', bbox_inches='tight')
    plt.clf()
    # If FC > 10:, find the intersection
    if fold_change > 10:
        def difference(z):
            pdf1_val = weights[0] * (1/(np.sqrt(2*np.pi*original_variances[0]))) * np.exp(-0.5 * ((z-original_means[0])**2)/original_variances[0])
            pdf2_val = weights[1] * (1/(np.sqrt(2*np.pi*original_variances[1]))) * np.exp(-0.5 * ((z-original_means[1])**2)/original_variances[1])
            return pdf1_val - pdf2_val
            
            # Define a function that restricts the search space to values less than 10
        lower_bound = 0
        upper_bound = 10
        try:
            intersection = brentq(difference, lower_bound, upper_bound)
            print("Intersection:", intersection)
        except ValueError:
            print("No intersection found in the given range.")



# Compute MM for lineages
for l in lins:
    # Extract data
    data = adata.obs[adata.obs.lineage == l].pct_counts_gene_group__mito_transcript.values.reshape(-1, 1)
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    # 1. Fit a GMM with multiple initializations
    gmm = GaussianMixture(n_components=2, n_init=10).fit(scaled_data.reshape(-1, 1))
    # Convert means and variances back to original space
    original_means = scaler.inverse_transform(gmm.means_).flatten()
    original_variances = gmm.covariances_.flatten() * (scaler.scale_**2)
    # 2. Calculate fold change
    fold_change = max(original_means) / min(original_means)
    log_fold_change = np.log2(fold_change)
    print("Fold Change:", fold_change)
    print("Log2 Fold Change:", log_fold_change)
    # 3. Plot the data
    x = np.linspace(min(data), max(data), 1000)
    scaled_x = scaler.transform(x.reshape(-1, 1)).flatten()
    weights = gmm.weights_
    pdf1 = weights[0] * (1/(np.sqrt(2*np.pi*original_variances[0]))) * np.exp(-0.5 * ((x-original_means[0])**2)/original_variances[0])
    pdf2 = weights[1] * (1/(np.sqrt(2*np.pi*original_variances[1]))) * np.exp(-0.5 * ((x-original_means[1])**2)/original_variances[1])
    total_pdf = pdf1 + pdf2
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    plt.plot(x, pdf1, label='Distribution 1')
    plt.plot(x, pdf2, label='Distribution 2')
    plt.plot(x, total_pdf, 'k--', label='Mixture Model')
    plt.legend()
    plt.title(f"{l}: MT percentage mixture models, FC={fold_change:.4f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(figpath + '/mixture_model_MT_perc_postqc_' + l + '.png', bbox_inches='tight')
    plt.clf()
    # If FC > 10:, find the intersection
    if fold_change > 4:
        def difference(z):
            pdf1_val = weights[0] * (1/(np.sqrt(2*np.pi*original_variances[0]))) * np.exp(-0.5 * ((z-original_means[0])**2)/original_variances[0])
            pdf2_val = weights[1] * (1/(np.sqrt(2*np.pi*original_variances[1]))) * np.exp(-0.5 * ((z-original_means[1])**2)/original_variances[1])
            return pdf1_val - pdf2_val
            
            # Define a function that restricts the search space to values less than 10
        lower_bound = 0
        upper_bound = 20
        try:
            intersection = brentq(difference, lower_bound, upper_bound)
            print("Intersection:", intersection)
        except ValueError:
            print("No intersection found in the given range.")

# Have a look at what would be lost at this res
immune_only = adata[adata.obs.lineage == "Immune"]
immune_only.obs['Immune_gt_MM_int'] = immune_only.obs['pct_counts_gene_group__mito_transcript'] > intersection
immune_only.obs['Immune_gt_MM_int'] = immune_only.obs['pct_counts_gene_group__mito_transcript'] > intersection
immune_only.obs['Immune_gt_MM_int'] = immune_only.obs['Immune_gt_MM_int'].astype('category')
sc.pl.umap(immune_only, color='Immune_gt_MM_int', frameon=False , title="Above or below intersection of MM for MT%", save="_Immune_above_MT_mm.png")
sc.pl.umap(immune_only, color='category', frameon=False , title="Above or below intersection of MM for MT%", save="_Immune_category.png")
sc.pl.umap(immune_only, color='label', frameon=False , title="Above or below intersection of MM for MT%", save="_Immune_label.png")
np.unique(immune_only.obs['Immune_gt_MM_int'], return_counts=True)

# What if we have a look for 3 components at the lineage model
for l in lins:
    # Extract data
    data = adata.obs[adata.obs.lineage == l].pct_counts_gene_group__mito_transcript.values.reshape(-1, 1)
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    # 1. Fit a GMM with multiple initializations
    gmm = GaussianMixture(n_components=3, n_init=10).fit(scaled_data.reshape(-1, 1))
    # Convert means and variances back to original space
    original_means = scaler.inverse_transform(gmm.means_).flatten()
    original_variances = gmm.covariances_.flatten() * (scaler.scale_**2)
    # 2. Calculate fold change
    fold_change = max(original_means) / min(original_means)
    log_fold_change = np.log2(fold_change)
    print("Fold Change:", fold_change)
    print("Log2 Fold Change:", log_fold_change)
    # 3. Plot the data
    x = np.linspace(min(data), max(data), 1000)
    scaled_x = scaler.transform(x.reshape(-1, 1)).flatten()
    weights = gmm.weights_
    pdf1 = weights[0] * (1/(np.sqrt(2*np.pi*original_variances[0]))) * np.exp(-0.5 * ((x-original_means[0])**2)/original_variances[0])
    pdf2 = weights[1] * (1/(np.sqrt(2*np.pi*original_variances[1]))) * np.exp(-0.5 * ((x-original_means[1])**2)/original_variances[1])
    pdf3 = weights[2] * (1/(np.sqrt(2*np.pi*original_variances[2]))) * np.exp(-0.5 * ((x-original_means[2])**2)/original_variances[2])
    total_pdf = pdf1 + pdf2 + pdf3
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    plt.plot(x, pdf1, label='Distribution 1')
    plt.plot(x, pdf2, label='Distribution 2')
    plt.plot(x, pdf3, label='Distribution 2')
    plt.plot(x, total_pdf, 'k--', label='Mixture Model')
    plt.legend()
    plt.title(f"{l}: MT percentage mixture models (total=3), FC={fold_change:.4f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(figpath + '/mixture_model_MT_perc_postqc_' + l + '_3models.png', bbox_inches='tight')
    plt.clf()

# Plot marker genes
gene_symb = ["MZB1", "JCHAIN", "CD19, MS4A1, CD79A", "CD163", "APOE", "C1QA-C", "CD3D", "CD4", ]
gene_ens = []
for value in gene_symb:
    # Create a boolean mask for the condition
    mask = adata.var['gene_symbols'] == value  # Change 'Name' to the column you want to search in
    if mask.any():
        # Use the index where the condition is True as the row name
        row_name = adata.var.index[mask].tolist()
        # Append the row name(s) to the list
        gene_ens.append(row_name)
    else:
        gene_ens.append('NA')

for index, g in enumerate(gene_symb):
    try:
        #sc.pl.umap(immune_only[immune_only.obs.lineage == "Immune"], color=gene_ens[index], frameon=False , title=g, save="_" + g + "_Immune.png")
        sc.pl.umap(adata[adata.obs.category.isin(["Enterocyte", "Stem cells"])], color=gene_ens[index], frameon=False , title=g, save="_" + g + "_Stem_Enterocyte.png")
    except:
        print("Gene not found")

# Have a look at whether or not a low threshdold for absolute cutoff would have removed these cells
adata.obs['abs_nmads_less_2'] = adata.obs['abs_mt_perc_nmads'].values < 2
adata.obs['abs_nmads_less_2'] = adata.obs['abs_nmads_less_2'].astype('category')
colors = {'True': 'orange', 'False': 'Blue'}
sc.pl.umap(adata, color='abs_nmads_less_2', frameon=False ,save='_post_batch_post_sweep_abs_nmad_less_2.png', palette="Set1")

# Do intermixing cells have 0 MT%
adata.obs['MT_gt_0'] = adata.obs['pct_counts_gene_group__mito_transcript'] > 0
adata.obs['MT_gt_0'] = adata.obs['MT_gt_0'].astype('category')
sc.pl.umap(adata, color='MT_gt_0', frameon=False ,save='_post_batch_post_sweep_no_mt_perc.png', palette="Set1")


#################################################


# Still looks like one samples isn't integrating very well
# Plot with axis and find this one
umap = pd.DataFrame(adata.obsm['X_umap'])
umap.columns = ["UMAP_1", "UMAP_2"]
umap['Sample'] = adata.obs.convoluted_samplename.values
plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap['UMAP_1'], umap['UMAP_2'], s=0.1)
x_ticks = np.arange(min(umap['UMAP_1']),max(umap['UMAP_1']), 0.3)
plt.xticks(x_ticks)
y_ticks = np.arange(min(umap['UMAP_2']),max(umap['UMAP_2']), 0.3)
plt.yticks(y_ticks)
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.tick_params(axis='both', labelsize=3)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_sample_ticks.png", dpi=300, bbox_inches='tight')
plt.clf()

# Plot this non-integrating sample
umap_x_min = 4.35
umap_x_max = 5.25
umap_y_min = 6.38
umap_y_max = 7.28
# Create a boolean mask based on the UMAP coordinates
umap_mask = (
    (adata.obsm['X_umap'][:, 0] >= umap_x_min) & (adata.obsm['X_umap'][:, 0] <= umap_x_max) &
    (adata.obsm['X_umap'][:, 1] >= umap_y_min) & (adata.obsm['X_umap'][:, 1] <= umap_y_max)
)
# Subset the AnnData object based on the mask
subset_adata = adata[umap_mask]
# Plot, coloured by sample
sc.pl.umap(subset_adata, color="convoluted_samplename", frameon=True, save="_post_batch_post_sweep_subset_coords1_samples.png", palette = 'Set3')
candidates = ["OTARscRNA13781777", "OTARscRNA13781786"]
subset_adata.obs['candidate_bad'] = adata.obs['convoluted_samplename'].apply(lambda x: x if x in candidates else 'No')
# Plot these ones only 
sc.pl.umap(subset_adata, color="candidate_bad", frameon=True, save="_post_batch_post_sweep_subset_coords1_samples_cols.png")
problem_sample = "OTARscRNA13781777"
# Plot UMAP of all samples, with this one highlighted
adata.obs['prob_bad'] = adata.obs['convoluted_samplename'].apply(lambda x: x if x in problem_sample else 'No')
sc.pl.umap(adata, color="c", frameon=True, save="_post_batch_post_sweep_bad.png")


# Looks like this sample has very few cells, and mostly immune - check this
np.unique(adata.obs.prob_bad, return_counts=True)
cats = np.unique(adata.obs.category)
count_cat = pd.DataFrame(np.unique(adata.obs[adata.obs.convoluted_samplename == problem_sample].category, return_counts=True)).T
count_cat.columns = ['label', 'freq']
plt.figure(figsize=(8, 6))
plt.pie(count_cat.freq, labels = count_cat.label, autopct='%.0f%%')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_bad_samples_prop.png", dpi=300, bbox_inches='tight')

# Check a random for categories
random_samps = np.random.choice(np.unique(adata.obs.convoluted_samplename), 3)
for index,s in enumerate(random_samps):
    count_cat = pd.DataFrame(np.unique(adata.obs[adata.obs.convoluted_samplename == s].category, return_counts=True)).T
    count_cat.columns = ['label', 'freq']
    plt.figure(figsize=(8, 6))
    plt.pie(count_cat.freq, labels = count_cat.label, autopct='%.0f%%')
    plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_random_good_samples_prop" + str(index) + ".png", dpi=300, bbox_inches='tight')



# Plot within category
cats = np.unique(adata.obs.category)
for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="label", title=c, frameon=False , save="_" + c + "_post_batch_post_sweep_keras.png")


# Also plot ngenes, n reads
sc.pl.umap(adata, color="ngenes_per_cell", frameon=True, save="_post_batch_post_sweep_ngenes.png")
# And total counts
sc.pl.umap(adata, color="total_counts", frameon=True, save="_post_batch_post_sweep_ncounts.png")
# And MT%
sc.pl.umap(adata, color="pct_counts_gene_group__mito_transcript", frameon=True, save="_post_batch_post_sweep_mtperc.png")

# Check categories, coloured by MT%
cats = np.unique(adata.obs.category)
def mad(data):
    return np.median(np.abs(data - np.median(data)))

for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="pct_counts_gene_group__mito_transcript", title=c, frameon=False , save="_" + c + "_post_batch_post_sweep_mtperc.png")
    # Print the median + 2.5*the median absolute deviation for each celltype
    #mt_perc=temp.obs["pct_counts_gene_group__mito_transcript"]
    #median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
    #print("For {}, the median={} and the median+2.5*MAD={}".format(c, np.median(mt_perc), median_plus_2pt5_mad))


# Is there a significant proportion of MT%, RP% and HLA% in the HVGs?
hvg = adata.var[adata.var.highly_variable == True]
mt_perc = hvg[hvg['gene_symbols'].str.contains("MT-")].shape[0]/hvg.shape[0]

# If we grouped cells into their lineages, what would relative cut off for MT perc be?
lin_df = pd.DataFrame({'category': cats, 'lineage': ""}) 
lin_df['lineage'] = np.where(lin_df['category'].isin(['B Cell', 'B Cell plasma', 'T Cell', 'Myeloid']), 'Immune', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category'].isin(['Stem cells', 'Secretory', 'Enterocyte']), 'Epithelial', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category']== 'Mesenchymal', 'Mesenchymal', lin_df['lineage'])
# Merge onto adata
adata.obs.index = adata.obs.index.astype(str)
cells = adata.obs.index
adata.obs = adata.obs.merge(lin_df, on='category', how="left")
adata.obs.index = cells

# Plot this on umap
sc.pl.umap(adata, color="lineage",frameon=True, save="_post_batch_post_sweep_lineage.png")


# Plot the MT% per category, adding a line for the relative cut off
for index,c in enumerate(cats):
    print(c)
    temp = adata[adata.obs.category == c]
    mt_perc=temp.obs["pct_counts_gene_group__mito_transcript"]
    median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
    print(median_plus_2pt5_mad)
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(mt_perc, hist=False, rug=True, label=c)
    if median_plus_2pt5_mad < 50:
        plt.axvline(median_plus_2pt5_mad, color=sns.color_palette()[cats.tolist().index(c)], linestyle='--')
    else:
        plt.axvline(50, color=sns.color_palette()[cats.tolist().index(c)], linestyle='--')


plt.legend()
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(data_name + "/" + status + "/" + category + "/figures" + '/postQC_relative_mt_perc_per_cat.png', bbox_inches='tight')
plt.clf()

# Now do this per lineage
lins = np.unique(lin_df.lineage)
for index,c in enumerate(lins):
    print(c)
    temp = adata[adata.obs.lineage == c]
    mt_perc=temp.obs["pct_counts_gene_group__mito_transcript"]
    median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
    print(median_plus_2pt5_mad)
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(mt_perc, hist=False, rug=True, label=c)
    if median_plus_2pt5_mad < 50:
        plt.axvline(median_plus_2pt5_mad, color=sns.color_palette()[lins.tolist().index(c)], linestyle='--')
    else:
        plt.axvline(50, color=sns.color_palette()[cats.tolist().index(c)], linestyle='--')
    nlost = len(mt_perc[mt_perc > median_plus_2pt5_mad])
    print("The number of cells lost in the {} lineage if we use this cut off would be {}".format(c, nlost))


plt.legend()
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(data_name + "/" + status + "/" + category + "/figures" + '/postQC_relative_mt_perc_per_lineage.png', bbox_inches='tight')
plt.clf()


# It is possible that the mixing is occuring between populations that are of low confidence (compared with the TI)
# Plot this divided by category
for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="Keras:predicted_celltype_probability", title=c, frameon=False , save="_" + c + "_post_batch_post_sweep_keras_conf.png")





# Have a look at the ngenes / cell / sample
samp_data = np.unique(adata.obs.convoluted_samplename, return_counts=True)
cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
high_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
bad_samps = problem_sample
depth_count = pd.DataFrame(index = np.unique(adata.obs.convoluted_samplename), columns=["Mean_nCounts", "nCells", "High_cell_sample", "ngenes_per_cell", "sample"])
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,1] = adata.obs[adata.obs.convoluted_samplename == samp].shape[0]
    depth_count.iloc[s,0] = sum(adata.obs[adata.obs.convoluted_samplename == samp].total_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,3] = sum(adata.obs[adata.obs.convoluted_samplename == samp].ngenes_per_cell)/depth_count.iloc[s,1]
    if samp in high_samps:
        depth_count.iloc[s,2] = "Red"
        depth_count.iloc[s,4] = samp
    else: 
         depth_count.iloc[s,2] = "Navy"
    # Also annotate the samples with low number of cells - Are these sequenced very deeply?
    if samp in bad_samps:
        depth_count.iloc[s,2] = "Green"
        depth_count.iloc[s,4] = samp

depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))
# Plot again
plt.figure(figsize=(8, 6))
plt.scatter(depth_count["Mean_nCounts"], depth_count["ngenes_per_cell"],  c=depth_count["High_cell_sample"], alpha=0.7)
plt.xlabel('Mean counts / cell')
plt.ylabel('Mean genes detected / cell')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_post_sweep_counts_cells_genes_cells_labels.png", bbox_inches='tight')
plt.clf()

# Look at the distribution of log10genes detected for bad samples and all samples
# Also save max density onto the table to look at
depth_count[["max_density"]] = 0
samps = np.unique(cells_sample["sample"])
for index,s in enumerate(samps):
    data = adata.obs[adata.obs.convoluted_samplename == s].ngenes_per_cell
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    if s in np.hstack((np.array(bad_samps, dtype="object"), bad_samps)):
        sns.distplot(data, hist=False, rug=True, label=s, kde_kws={'linewidth': 2.5})
    else:
        sns.distplot(data, hist=False, rug=True, color='black')
    # Find the max value (remove this code if just want to plot the problem ones)
    kde = sns.kdeplot(data, shade=True)
    depth_count.iloc[index,6] = kde.get_lines()[index].get_ydata().max()

plt.legend()
plt.xlabel('nGenes detected/cell')
ax.set(xlim=(0, max(adata.obs.ngenes_per_cell)))
# Set y-axis limits to match the maximum density
ax.set_ylim(0, 0.0017)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_genes_per_sample.png", bbox_inches='tight')
plt.clf()

# Have a look at the ones with very high low ngenes/cell
# cut off at 0.0008 (initially, removed the very high cell number, bad quality samples)
# Now looking like 0.0006 may capture additional, badly integrating samples
cutoff=0.0006
potbadsamps = depth_count[depth_count.max_density > cutoff]
potbadsamps = np.array(potbadsamps.index)
for index,s in enumerate(samps):
    data = adata.obs[adata.obs.convoluted_samplename == s].ngenes_per_cell
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    if s in potbadsamps:
        sns.distplot(data, hist=False, rug=True, label=s, kde_kws={'linewidth': 2.5})
    else:
        sns.distplot(data, hist=False, rug=True, color='black')

plt.legend()
plt.xlabel('nGenes detected/cell')
ax.set(xlim=(0, max(adata.obs.ngenes_per_cell)))
# Set y-axis limits to match the maximum density
ax.set_ylim(0, 0.0017)
plt.axhline(y = cutoff, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_genes_per_potential_bad_sample.png", bbox_inches='tight')
plt.clf()

# Plot these on UMAP
adata.obs['potential_bad_sample'] = adata.obs['convoluted_samplename'].apply(lambda x: x if x in potbadsamps else 'No')
sc.pl.umap(adata, color="potential_bad_sample",frameon=True, save="_post_batch_post_sweep_low_n_gene_cell_samples.png")

# Check the ncells/sample, some of these look like they have very few cells
depth_count[depth_count.index.isin(potbadsamps)]

# Plot the other potential bad samples in the Mean genes detect/cell
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    if samp in high_samps:
        depth_count.iloc[s,2] = "Red"
        depth_count.iloc[s,4] = samp
    else: 
         depth_count.iloc[s,2] = "Navy"
    # Also annotate the samples with low number of cells - Are these sequenced very deeply?
    if samp in potbadsamps:
        depth_count.iloc[s,2] = "Green"
        depth_count.iloc[s,4] = samp

# Plot again
plt.figure(figsize=(8, 6))
plt.scatter(depth_count["Mean_nCounts"], depth_count["ngenes_per_cell"],  c=depth_count["High_cell_sample"], alpha=0.7)
plt.xlabel('Mean counts / cell')
plt.ylabel('Mean genes detected / cell')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_post_sweep_counts_cells_genes_cells_labels_potbad.png", bbox_inches='tight')
plt.clf()


# Determining new cut off
depth_count[["cross_density"]] = 0
for index,s in enumerate(samps):
    data = adata.obs[adata.obs.convoluted_samplename == s].ngenes_per_cell
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    if s in potbadsamps:
        sns.distplot(data, hist=False, rug=True, label=s, kde_kws={'linewidth': 2.5})
    else:
        sns.distplot(data, hist=False, rug=True, color='black')
    # Find the max value (remove this code if just want to plot the problem ones)
    kde = sns.kdeplot(data, shade=False)
    ax = plt.gca()
    x_values, y_values = ax.lines[index].get_data()
    # Get x when y crosses
    x_thresh = x_values[y_values >= cutoff]
    # Take the last of these
    if len(x_thresh > 0):
        threshold_x = x_thresh[len(x_thresh)-1]
        print(threshold_x)
        depth_count.iloc[index,7] = threshold_x

plt.legend()
plt.xlabel('nGenes detected/cell')
ax.set(xlim=(0, max(adata.obs.ngenes_per_cell)))
# Set y-axis limits to match the maximum density
ax.set_ylim(0, 0.0017)
plt.axhline(y = cutoff, color = 'red', linestyle = '--', alpha = 0.5)
plt.axvline(x = max(depth_count.cross_density), color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_genes_per_potential_bad_sample_newthresh.png", bbox_inches='tight')
plt.clf()
print(max(depth_count.cross_density))
depth_count.cross_density[depth_count.cross_density > 0]


# Define cells as above or below a cut off and plot
ngene_cutoff = 300
adata.obs['low_ngene_cell'] = adata.obs['ngenes_per_cell'].apply(lambda x: 'Yes' if x < ngene_cutoff else 'No')
sc.pl.umap(adata, color="low_ngene_cell",frameon=True, save="_post_batch_post_sweep_low_n_gene_cell_low_ngene_cell.png")

# Save a list of the potentially bad samples - These will be removed in the next round of integration and analysis
np.savetxt(data_name + "/" + status + "/" + category + '/bad_samples_to_remove.txt', potbadsamps, delimiter='\n', fmt='%s')


# Save umap version
#umap_file = 

# Plot umap with dimensions
umap = pd.DataFrame(adata.obsm['X_umap'])
umap.columns = ["UMAP_1", "UMAP_2"]
umap['Sample'] = adata.obs.convoluted_samplename.values
plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap['UMAP_1'], umap['UMAP_2'], s=0.1, c=np.arange(len(umap['Sample'])), cmap='Set1')
x_ticks = np.arange(min(umap['UMAP_1']),max(umap['UMAP_1']), 0.3)
plt.xticks(x_ticks)
y_ticks = np.arange(min(umap['UMAP_2']),max(umap['UMAP_2']), 0.3)
plt.yticks(y_ticks)
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.tick_params(axis='both', labelsize=3)
plt.colorbar(scatter, label='Sample')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_sample_ticks.png", dpi=300, bbox_inches='tight')
plt.clf()


# Plot within category
cats = np.unique(adata.obs.category)
for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="label", title=c,frameon=False, figsize=(6,4), save="_" + c + "_post_batch_post_sweep_keras.png")

# Rename convoluted sample name (this seems to have been lost after the batch correction). 
# Have checked the adata used for batch correction and this is correct though. So not to worry.
#adata.obs["convoluted_samplename"] = adata.obs['convoluted_samplename'].str.replace("__donor", '')

# Check where the high samples look here
samp_data = np.unique(adata.obs.convoluted_samplename, return_counts=True)
cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
high_cell_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
def check_intersection(value):
    if value in high_cell_samps:
        return 'yes'
    else:
        return 'no'

# Create a new column 'intersects' using the apply method
adata.obs['high_cell_samp'] = adata.obs['convoluted_samplename'].apply(lambda x: check_intersection(x))

# Plot the high cell samples labelled by their ID
def fill_column(row):
    if row['high_cell_samp'] == 'yes':
        return row['convoluted_samplename']
    else:
        return 'Not high sample'  # You can also specify a different default value if needed

# Create a new column 'NewColumn' based on the conditions
adata.obs['high_cell_samp_label'] = adata.obs.apply(fill_column, axis=1)
# Plot these
sc.pl.umap(adata, color="high_cell_samp_label", frameon=True, save="_post_batch_post_sweep_high_samples.png")

# Subset adata for problem region to try and identify the non-integrating samples
umap_x_min = 0.4
umap_x_max = 1
umap_y_min = 3.48
umap_y_max = 3.7
# Create a boolean mask based on the UMAP coordinates
umap_mask = (
    (adata.obsm['X_umap'][:, 0] >= umap_x_min) & (adata.obsm['X_umap'][:, 0] <= umap_x_max) &
    (adata.obsm['X_umap'][:, 1] >= umap_y_min) & (adata.obsm['X_umap'][:, 1] <= umap_y_max)
)
# Subset the AnnData object based on the mask
subset_adata = adata[umap_mask]
# Plot, coloured by sample
sc.pl.umap(subset_adata, color="convoluted_samplename", frameon=True, save="_post_batch_post_sweep_subset_coords1_samples.png")

# Have a look at the potentially bad samples (inc. high cell number samples)



# Have a look at the high cell samples seperately and seperately within a couple categories
high = adata[adata.obs.high_cell_samp == "yes"]
sc.pl.umap(high, color="category",title="High cell samples", frameon=False, save="_post_batch_post_sweep_high_cell_number.png")
good = adata[adata.obs.high_cell_samp == "no"]
sc.pl.umap(good, color="category",title="Not high cell samples", frameon=False, save="_post_batch_post_sweep_not_high_cell_number.png")
# For just the B Cells / T Cells
for i, val in enumerate(["B Cell", "T Cell"]):
    temp = adata[adata.obs.category == val]
    high = temp[temp.obs.high_cell_samp == "yes"]
    good = temp[temp.obs.high_cell_samp == "no"]
    sc.pl.umap(high, color="label",title="High cell samples - " + val, frameon=False, save="_" + val  + "_post_batch_post_sweep_high_cell_number.png")
    sc.pl.umap(good, color="label",title="Not high cell samples - " + val, frameon=False, save="_" + val  + "_post_batch_post_sweep_not_high_cell_number.png")






# Assessment of batch effect integration
df = pd.read_csv(data_name + "/" + status + "/" + category + "/tables/integration_benchmarking.csv")



