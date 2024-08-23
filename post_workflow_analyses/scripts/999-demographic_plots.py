# Bradley August 2024
# Plotting demographics of the IBDverse cohort

# Load libraries
import scanpy as sc
import pandas as pd
import numpy as np
import os
from anndata.experimental import read_elem
from h5py import File
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotnine as plt9
from matplotlib.lines import Line2D

# Load the input anndata (this contains the expression data actually used in the cohort)
f = "input_v7/adata_raw_all_tissues.h5ad"
f2 = File(f, 'r')
# Read only cell-metadata
obs = read_elem(f2['obs'])

# Get the metadata and subset for samples we definitely used
meta = pd.read_csv("input_v7/samples_select_atlas_brad.tsv", sep = "\t")
meta = meta[meta['sanger_sample_id'].isin(obs['sanger_sample_id'])]

# Define outdir
outdir = "other_paper_figs"
if os.path.exists(outdir) == False:
        os.mkdir(outdir)

# Plot stacked bar plots of several different types of demographic data and distribution across them
meta.rename(columns={"biopsy_type": "tissue"}, inplace=True)
meta['tissue']=meta['tissue'].replace(["r", "blood", "ti"],  ["Rectum", "Blood", "TI"])
meta['ses_cd_binned'] = meta['ses_cd_binned'].astype(str)
meta['ses_cd_binned'] = meta['ses_cd_binned'].replace("nan", "Missing")
to_plot = {"tissue": ["disease_status", "sex", "ses_cd_binned", "inflammation_status"]}
for v in to_plot:
    df_melted = pd.melt(meta, id_vars=v, value_vars=to_plot[v], var_name='category', value_name='value')
    unique_values = df_melted['value'].unique()
    palette = sns.color_palette("Dark2", len(unique_values))
    color_mapping = dict(zip(unique_values, palette))
    df_grouped = df_melted.groupby(['tissue', 'category', 'value']).size().reset_index(name='count')
    df_grouped['proportion'] = df_grouped.groupby(['tissue', 'category'])['count'].transform(lambda x: x / x.sum())
    df_pivot = df_grouped.pivot_table(index=['tissue', 'category'], columns='value', values='proportion', fill_value=0)
    categories = df_pivot.index.get_level_values('category').unique()
    palette = sns.color_palette("Dark2", len(df_grouped['value'].unique()))
    fig, axes = plt.subplots(1, len(categories), figsize=(len(categories) * 5, 6), sharey=True)
    df_pivot.reset_index(inplace=True)
    for i, category in enumerate(categories):
        ax = axes[i]
        df_category = df_pivot[df_pivot['category'] == category]
        df_category.set_index("tissue", inplace=True)
        df_category.drop(columns=["category"], inplace=True)
        # Plot stacked bar chart
        df_category_colors = [color_mapping[val] for val in df_category.columns]
        df_category.plot(kind='bar', stacked=True, ax=ax, color=df_category_colors)
        # Set titles and labels
        ax.set_title(category)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion" if i == 0 else "")
        ax.legend_.remove()
    #
    # Adjust layout
    legend_elements = []
    for category in categories:
        # Add subtitle for the category
        legend_elements.append(Line2D([0], [0], color='w', marker='None', label=f"{category}:", linestyle='None', markerfacecolor='None'))
        # Add legend entries for each unique value in the category
        unique_vals = df_grouped[df_grouped['category'] == category]['value'].unique()
        for val in unique_vals:
            legend_elements.append(Line2D([0], [0], color=color_mapping[val], lw=4, label=val))
    # Place the legend outside the plot
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, title="")
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{outdir}/demographics_across_{v}.png", bbox_inches='tight')
    plt.clf()
