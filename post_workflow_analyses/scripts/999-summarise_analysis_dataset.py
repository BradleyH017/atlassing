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
from scipy import stats
 


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
axmax = max([max(prop_plt['high_qc_proportion']), max(prop_plt['prediction_proportion'])])*1.1
fig, ax = plt.subplots(figsize=(8, 6))
rho, p_value = stats.pearsonr(prop_plt['high_qc_proportion'], prop_plt['prediction_proportion'])
fig, ax = plt.subplots(figsize=(8, 6))
for lineage in prop_plt['manual_lineage'].unique():
    subset = prop_plt[prop_plt['manual_lineage'] == lineage]
    ax.scatter(subset['high_qc_proportion'], subset['prediction_proportion'],
               label=lineage)

ax.plot([0, 1], [0, 1], ls='--', color='black', label='x = y')
ax.set_xlim(0, axmax)
ax.set_ylim(0, axmax)
x_vals = np.array(ax.get_xlim())
y_vals = rho * x_vals 
ax.plot(x_vals, y_vals, '--', color='red', label=f'rho = {rho:.2f}\np-value = {p_value:.2e}')
ax.set_xlabel('High QC Proportion')
ax.set_ylabel('Prediction Proportion')
plt.legend()
ax.set_title('Concordance of high QC and predicted labels')
plt.savefig(f"{outdir}/highqc_vs_predicted_proportions.png", bbox_inches='tight')
plt.clf()

# Have a look at the biggest changers
prop_plt['pred_over_highqc'] = prop_plt['prediction_proportion'] / prop_plt['high_qc_proportion']


######## 5. Are the over-estimated clusters associated with cells/sample of lower quality? #########
# Look at the distribution of QC metrics in these clusters compared to other epithelial
obs['log_n_genes_by_counts'] = np.log10(obs['n_genes_by_counts'])
obs['log_total_counts'] = np.log10(obs['total_counts'])
obs['manual_lineage'] = obs["predicted_labels"].apply(lambda x: x.split('_')[0] if '_' in x else x)
epiobs = obs[obs['manual_lineage'] == "Epithelial"]
cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
clusters = np.unique(epiobs['predicted_labels'])
problem = ["Epithelial_6", "Epithelial_8"]
for c in cols:
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for k in clusters:
        print(k)
        data = epiobs.loc[epiobs['predicted_labels'] == k, c].values
        if k in problem:
            sns.distplot(data, hist=False, rug=True, label=k)
        else:
            sns.distplot(data, hist=False, rug=True, kde_kws={'color': 'orange'})
    #
    plt.legend()
    plt.xlabel(c)
    plt.title(f"Distribution of {c} across clusters")
    plt.savefig(f"temp/Distribution_of_{c}_across_epithelial_clusters.png", bbox_inches='tight')
    plt.clf()

# If we subset cells for those that pass the initial filtering, how do the contribution of cells to these clusters change? 
obs_filt = obs[(obs['pct_counts_gene_group__mito_transcript'] < 50) & (obs['log_n_genes_by_counts'] > np.log10(250)) & (obs['log_total_counts'] > np.log10(500))]
pred_freq = pd.DataFrame(obs['predicted_labels'].value_counts()).reset_index(names="leiden")
pred_freq.rename(columns={"count":"predicted_freq"},inplace=True)
pred_freq_filt = pd.DataFrame(obs_filt['predicted_labels'].value_counts()).reset_index(names="leiden")
pred_freq_filt.rename(columns={"count":"predicted_filt_freq"},inplace=True)
freq_plt = pred_freq.merge(pred_freq_filt, how="left", on="leiden")
freq_plt['prop_after_filt'] = freq_plt['predicted_filt_freq'] / freq_plt['predicted_freq']
freq_plt['manual_lineage'] = freq_plt["leiden"].apply(lambda x: x.split('_')[0] if '_' in x else x)
for l in lins:
    temp = freq_plt[freq_plt['manual_lineage'] == l]
    plt.bar(temp['leiden'], temp['prop_after_filt'])
    plt.xlabel('Cluster')
    plt.ylabel('Proportion of predicted cells kept after filtering')
    plt.title('Effect of first round filtration on the predicted cell-type proportions')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f"temp/Prop_predicted_{l}_after_filtration.png", bbox_inches='tight')
    plt.clf()

# Plot proportion of predicted cells vs original after filtering
pred_props_filt = pd.DataFrame(obs_filt['predicted_labels'].value_counts(normalize=True)).reset_index(names="leiden")
pred_props_filt.rename(columns={"proportion": "prediction_proportion_filtered"}, inplace=True)
prop_plt = prop_plt.merge(pred_props_filt, on="leiden", how="left")
prop_plt['filt_over_pred'] = prop_plt['prediction_proportion_filtered'] / prop_plt['prediction_proportion']
prop_plt['filt_pred_over_hqc'] = prop_plt['prediction_proportion_filtered'] / prop_plt['high_qc_proportion']
axmax = max([max(prop_plt['high_qc_proportion']), max(prop_plt['prediction_proportion_filtered'])])*1.1
fig, ax = plt.subplots(figsize=(8, 6))
rho, p_value = stats.pearsonr(prop_plt['high_qc_proportion'], prop_plt['prediction_proportion_filtered'])
for lineage in prop_plt['manual_lineage'].unique():
    subset = prop_plt[prop_plt['manual_lineage'] == lineage]
    ax.scatter(subset['high_qc_proportion'], subset['prediction_proportion_filtered'],
               label=lineage)

ax.plot([0, 1], [0, 1], ls='--', color='black', label='x = y')
#ax.axhline(y=rho, color='red', linestyle='--', label=f'rho = {rho:.2f}\np-value = {p_value:.2e}')
x_vals = np.array(ax.get_xlim())
y_vals = rho * x_vals 
ax.plot(x_vals, y_vals, '--', color='red', label=f'rho = {rho:.2f}\np-value = {p_value:.2e}')
ax.set_xlim(0, axmax)
ax.set_ylim(0, axmax)
ax.set_xlabel('High QC Proportion')
ax.set_ylabel('Prediction Proportion after filtering')
plt.legend()
ax.set_title('Concordance of high QC and predicted labels after filtering')
plt.savefig(f"{outdir}/highqc_vs_predicted_proportions_postfilt.png", bbox_inches='tight')
plt.clf()

# How many of the cells gained by prediction are from this cluster? 
hqc_freq = pd.DataFrame(hqc_obs['leiden'].value_counts()).reset_index()
epi8_hqc = hqc_freq.loc[hqc_freq['leiden'] == "Epithelial_8", "count"].values[0]
epi8_pred_filt = freq_plt.loc[freq_plt['leiden'] == "Epithelial_8", "predicted_filt_freq"].values[0]
gain_epi8 = epi8_pred_filt-epi8_hqc
gain_all = obs_filt.shape[0] - hqc_obs.shape[0]
gain_epi8 / gain_all

######## Have a look at the distribution of confidence for each annotation #########
for l in lins:
    print(l)
    temp = obs[obs['manual_lineage'] == l]
    clusters = np.unique(temp['predicted_labels'])
    fig, axes = plt.subplots(nrows=len(clusters), ncols=1, figsize=(8, 1 * len(clusters)), sharex=True)
    plt.title(f"Distribution of conf_score across clusters")
    for i, k in enumerate(clusters):
        print(k)
        if len(clusters) > 1:
            ax = axes[i]
        else:
            ax = axes
        data = temp.loc[temp['predicted_labels'] == k, "conf_score"].values
        sns.distplot(data, hist=False, rug=True, ax=ax)
        ax.text(0.95, 0.95, f'Cluster: {k}', transform=ax.transAxes,fontsize=12, verticalalignment='top', horizontalalignment='right')
    # Set labels and title for each subplot
    ax.set_xlabel('CellTypist conf_score')
    plt.savefig(f"temp/Distribution_of_conf_score_across_{l}_clusters.png", bbox_inches='tight')
    plt.clf()
    

###### What if we just apply a threshold of MT% < 60
obs_filt = obs[(obs['pct_counts_gene_group__mito_transcript'] < 60)]
pred_freq = pd.DataFrame(obs['predicted_labels'].value_counts()).reset_index(names="leiden")
pred_freq.rename(columns={"count":"predicted_freq"},inplace=True)
pred_freq_filt = pd.DataFrame(obs_filt['predicted_labels'].value_counts()).reset_index(names="leiden")
pred_freq_filt.rename(columns={"count":"predicted_filt_freq"},inplace=True)
freq_plt = pred_freq.merge(pred_freq_filt, how="left", on="leiden")
freq_plt['prop_after_filt'] = freq_plt['predicted_filt_freq'] / freq_plt['predicted_freq']
freq_plt['manual_lineage'] = freq_plt["leiden"].apply(lambda x: x.split('_')[0] if '_' in x else x)
for l in lins:
    temp = freq_plt[freq_plt['manual_lineage'] == l]
    plt.bar(temp['leiden'], temp['prop_after_filt'])
    plt.xlabel('Cluster')
    plt.ylabel('Proportion of predicted cells kept after filtering')
    plt.title('Effect of first round filtration on the predicted cell-type proportions')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f"temp/Prop_predicted_{l}_after_filtration_mt_only.png", bbox_inches='tight')
    plt.clf()

pred_props_filt = pd.DataFrame(obs_filt['predicted_labels'].value_counts(normalize=True)).reset_index(names="leiden")
pred_props_filt.rename(columns={"proportion": "prediction_proportion_filtered"}, inplace=True)
prop_plt = prop_plt.merge(pred_props_filt, on="leiden", how="left")
prop_plt['filt_over_pred'] = prop_plt['prediction_proportion_filtered'] / prop_plt['prediction_proportion']
prop_plt['filt_pred_over_hqc'] = prop_plt['prediction_proportion_filtered'] / prop_plt['high_qc_proportion']
axmax = max([max(prop_plt['high_qc_proportion']), max(prop_plt['prediction_proportion_filtered'])])*1.1
fig, ax = plt.subplots(figsize=(8, 6))
rho, p_value = stats.pearsonr(prop_plt['high_qc_proportion'], prop_plt['prediction_proportion_filtered'])
for lineage in prop_plt['manual_lineage'].unique():
    subset = prop_plt[prop_plt['manual_lineage'] == lineage]
    ax.scatter(subset['high_qc_proportion'], subset['prediction_proportion_filtered'],
               label=lineage)


ax.plot([0, 1], [0, 1], ls='--', color='black', label='x = y')
#ax.axhline(y=rho, color='red', linestyle='--', label=f'rho = {rho:.2f}\np-value = {p_value:.2e}')
x_vals = np.array(ax.get_xlim())
y_vals = rho * x_vals 
ax.plot(x_vals, y_vals, '--', color='red', label=f'rho = {rho:.2f}\np-value = {p_value:.2e}')
ax.set_xlim(0, axmax)
ax.set_ylim(0, axmax)
ax.set_xlabel('High QC Proportion')
ax.set_ylabel('Prediction Proportion after filtering')
plt.legend()
ax.set_title('Concordance of high QC and predicted labels after filtering')
plt.savefig(f"{outdir}/highqc_vs_predicted_proportions_postfilt_mt_only.png", bbox_inches='tight')
plt.clf()

# Are the remaining cells of lower quality still? 
epiobs_filt = obs_filt[obs_filt['manual_lineage'] == "Epithelial"]
cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
clusters = np.unique(epiobs_filt['predicted_labels'])
problem = ["Epithelial_6", "Epithelial_8"]
for c in cols:
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for k in clusters:
        print(k)
        data = epiobs_filt.loc[epiobs_filt['predicted_labels'] == k, c].values
        if k in problem:
            sns.distplot(data, hist=False, rug=True, label=k)
        else:
            sns.distplot(data, hist=False, rug=True, kde_kws={'color': 'orange'})
    #
    plt.legend()
    plt.xlabel(c)
    plt.title(f"Distribution of {c} across clusters")
    plt.savefig(f"temp/Distribution_of_{c}_across_epithelial_clusters_mt_filt_only.png", bbox_inches='tight')
    plt.clf()
    
# Do this for all lineages:
for l in lins:
    print(l)
    epiobs_filt = obs_filt[obs_filt['manual_lineage'] == l]
    cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
    clusters = np.unique(epiobs_filt['predicted_labels'])
    problem = ["Epithelial_6", "Epithelial_8"]
    for c in cols:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for k in clusters:
            print(k)
            data = epiobs_filt.loc[epiobs_filt['predicted_labels'] == k, c].values
            if k in problem:
                sns.distplot(data, hist=False, rug=True, label=k)
            else:
                sns.distplot(data, hist=False, rug=True, kde_kws={'color': 'orange'})
        #
        plt.legend()
        plt.xlabel(c)
        plt.title(f"Distribution of {c} across clusters")
        plt.savefig(f"temp/Distribution_of_{c}_across_{l}_clusters_mt_filt_only.png", bbox_inches='tight')
        plt.clf()
    
# Find median MT% per cluster
for k in clusters:
    res = np.median(epiobs_filt.loc[epiobs_filt['predicted_labels'] == k, "pct_counts_gene_group__mito_transcript"].values)
    print(f"{k}: {res}")
    
    
######### What if we filter solely on nGenes (this is essentially the important the)

######## 6. Find the relationship between QC, CellTypist confidence #########
# As we increase celltypist confidence score, how many cells remaining set are in the lower quality set?
thresh = np.arange(0.5, 1.01, 0.02)
def get_testable_samps_clusters(obs, score_thresh, ncells_per_sample, nsamples):
    cells_sample_cluster = obs[obs['conf_score'] > score_thresh].groupby("predicted_labels")['sanger_sample_id'].value_counts().reset_index()
    cells_sample_cluster = cells_sample_cluster[cells_sample_cluster['count'] > ncells_per_sample]
    samples_per_cluster = cells_sample_cluster.groupby('predicted_labels').size().reset_index()
    ntestable_clusters = samples_per_cluster[samples_per_cluster[0] > nsamples].shape[0]
    med_samps_per_cluster = np.median(samples_per_cluster[0])
    return(ntestable_clusters, med_samps_per_cluster)

test_ncluster, test_nsamps = get_testable_samps_clusters(obs=obs, score_thresh = 0.7, ncells_per_sample=5, nsamples=50)
test_ncluster_filt, test_nsamps_filt = get_testable_samps_clusters(obs=obs_filt, score_thresh = 0.7, ncells_per_sample=5, nsamples=50)

res = []
for v in thresh:
    print(v)
    total=sum(obs['conf_score'] > v)
    nhigh=sum(obs_filt['conf_score'] > v)
    test_ncluster, test_nsamps = get_testable_samps_clusters(obs=obs, score_thresh = v, ncells_per_sample=5, nsamples=50)
    test_ncluster_filt, test_nsamps_filt = get_testable_samps_clusters(obs=obs_filt, score_thresh = v, ncells_per_sample=5, nsamples=50)
    temp = pd.DataFrame({"conf_score": [v], "total_passing": [total], "passing_high_qc": [nhigh], "n-eQTLable_clusters": [test_ncluster], "med_eQTLable_samps_per_cluster": [test_nsamps], "n-eQTLable_clusters_hqc": [test_ncluster_filt], "med_eQTLable_samps_per_cluster_hqc": [test_nsamps_filt]})
    res.append(temp)

thresh_all = pd.concat(res)
thresh_all['perc_high_qc'] = thresh_all['passing_high_qc'] / thresh_all['total_passing']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_all['conf_score'], thresh_all['total_passing'], label='Total Passing', marker='o', color='blue')
ax.plot(thresh_all['conf_score'], thresh_all['passing_high_qc'], label='Passing High QC', marker='o', color='orange')
ax.set_xlabel('CellTypist confidence score')
ax.set_ylabel('Number of cells')
ax.set_title('Total Passing CellTypist and Passing Cell Typist and round 1 QC')
ax.set_xlim(0.5, 1.0)
ax.axhline(y=1995772, color='red', linestyle='--', label=f'round2 output (pre-any prediction)')
ax.legend()
plt.savefig(f"temp/Conf_score_vs_cellQC.png", bbox_inches='tight')
plt.clf()

# Plot eQTLable
# TO DO: COmpare this to the set in the round2 high QC set
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_all['conf_score'], thresh_all['n-eQTLable_clusters'], label='Number of eQTL testable clusters', marker='o', color='darkred')
ax.plot(thresh_all['conf_score'], thresh_all['med_eQTLable_samps_per_cluster'], label='Median eQTL samples per cluster', marker='o', color='green')
ax.plot(thresh_all['conf_score'], thresh_all['n-eQTLable_clusters_hqc'], linestyle='--', color='darkred', marker='o')
ax.plot(thresh_all['conf_score'], thresh_all['med_eQTLable_samps_per_cluster_hqc'], linestyle='--', color='green', marker='o')
ax.set_xlabel('CellTypist confidence score')
ax.set_ylabel('Number of cells')
ax.set_title('eQTL testing potential vs CellTypist confidence score')
ax.set_xlim(0.5, 1.0)
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='black', linestyle='--', label='High QC only'))
ax.legend(handles=handles)
plt.savefig(f"temp/Conf_score_vs_eQTL_testing.png", bbox_inches='tight')
plt.clf()

######## 7. Update pie plot #########
df_grouped = obs_filt.groupby('predicted_labels').agg(
    nCells=('predicted_labels', 'size'),
    nSamples=('sanger_sample_id', 'nunique')
).reset_index()
df_grouped['log10_nCells'] = np.log10(df_grouped['nCells'])
df_proportions = obs_filt.groupby(['predicted_labels', 'tissue']).size().reset_index(name='count')
df_proportions['proportion'] = df_proportions.groupby('predicted_labels')['count'].transform(lambda x: x / x.sum())
tissue_types = obs_filt['tissue'].cat.categories
tissue_colors = dict(zip(tissue_types, Category10_10[:len(tissue_types)]))  # Map tissues to Category10 colors

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
texts = []
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
plt.savefig(f"{outdir}/prop_tissues_per_cluster_filtered.png", bbox_inches='tight')
plt.show()