##################################################
############# Bradley August 2023 ################
# conda activate sc4
# Analysis of the keras score predictions per cell
# and comparison of these to those of celltypist
##################################################

# Change dir
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
cwd = os.getcwd()
print(cwd)

# Load libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Define path options
data_name="rectum"
status="healthy"
category="All_not_gt_subset"
statpath = data_name + "/" + status
catpath = statpath + "/" + category
figpath = catpath + "/figures"
tabdir = catpath + "/tables"
objpath = catpath + "/objects"
sc.settings.figdir=figpath

# Load in the rectum object (post processing)data_name + "/" + status + "/" + category + "/objects/adata_objs_param_sweep"
adata = ad.read_h5ad(objpath + "/adata_objs_param_sweep/NN_350_scanvi_umap.adata")
samples = np.unique(adata.obs.experiment_id)
import re
def remove_char(lst, char):
    return [re.sub(char, '', s) for s in lst]

samples = remove_char(samples,"__donor")


# Load in the confidence for each cell from every sample (https://saturncloud.io/blog/efficiently-appending-to-a-dataframe-within-a-for-loop-in-python/)
dir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/2023_08_07/rectum/results/celltype/keras_celltype"
data = []
for i in range(0,len(samples)):
    print(samples[i])
    temp = ad.read_h5ad(dir + "/" + samples[i] + "/" + samples[i] + "___cellbender_fpr0pt1-scrublet-ti_freeze003_prediction-predictions.h5ad")
    want = temp.obs.columns.str.contains("probability__")
    temp_probs = temp.obs.loc[:,want]
    data.append(temp_probs)
    del temp, temp_probs

keras = pd.concat(data, ignore_index=False)
# Subset for those passing QC from the 'atlassing_rectum_all_samples.py' script
keras = keras[keras.index.isin(adata.obs.index)]

# For each cell, reduce to the first, second and third most confident hits
ktop3 = pd.DataFrame(index=keras.index, columns=["first", "first_confidence", "second", "second_confidence", "third", "third_confidence"])
for c in keras.index:
    temp = keras.loc[c,:]
    temp.sort_values(inplace = True, ascending=False)
    ktop3.loc[c,"first"] = temp.index[0]
    ktop3.loc[c,"first_confidence"] = temp[0]
    ktop3.loc[c,"second"] = temp.index[1]
    ktop3.loc[c,"second_confidence"] = temp[1]
    ktop3.loc[c,"third"] = temp.index[2]
    ktop3.loc[c,"third_confidence"] = temp[2]

# Save the ktop3
ktop3.to_csv(tabdir + "/keras_top3_postcellfilter.csv")


# Plot the proportion of the second versus the first annotation, within category and annotated by cell-type
compfigpath = figpath + "/anno_conf"
kerasconf = compfigpath + "/keras"
ctconf = compfigpath + "/celltypist"
if os.path.exists(compfigpath) == False:
    os.mkdir(compfigpath)

if os.path.exists(kerasconf) == False:
    os.mkdir(kerasconf)

if os.path.exists(ctconf) == False:
    os.mkdir(ctconf)

cats = np.unique(adata.obs.category)
for c in cats:
    labels = np.unique(adata.obs[adata.obs.category == c].label)
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for l in labels:
        idx = adata.obs.label == l
        dat = ktop3.first_confidence[idx] - ktop3.second_confidence[idx]
        sns.distplot(dat, hist=False, rug=True, label=l)
    plt.legend()
    ax.set(xlim=(0, 1))
    plt.title(c)
    plt.xlabel('1st - 2nd annotation confidence')
    plt.savefig(kerasconf + '/' + c + '_keras_second_over_first.png', bbox_inches='tight')
    plt.clf()


# Violin plot of the confidence for each group
ktop3[["first"]] = ktop3[["first"]].astype("category")
ktop3['first_confidence'] = ktop3['first_confidence'].astype('float')
ktop3['first'] = remove_char(ktop3['first'],"probability__")
ktop3['second'] = remove_char(ktop3['second'],"probability__")
for c in cats:
    labels = np.unique(adata.obs[adata.obs.category == c].label__machine)
    temp = ktop3[ktop3['first'].isin(labels)]
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    sns.violinplot(data=temp, x="first", y="first_confidence", order=labels)
    ax.set_xticklabels(labels = labels, rotation=45, horizontalalignment='right')
    ax.set(ylim=(0, 1))
    plt.title(c)
    plt.savefig(kerasconf + '/' + c + '_keras_first_anno_confidence.png', bbox_inches='tight')
    plt.clf()


# Plotting the calls of second cells versus the first - Is there a common pattern?
fs = ktop3.groupby(['first', 'second']).size().reset_index(name='frequency')
fs['prop'] = 0
catnum = np.unique(ktop3['first'], return_counts = True)
for i in range(0, fs.shape[0]):
    fcat = fs['first'][i]
    catidx = np.where(catnum[0] == fcat)
    cattot = catnum[1][np.where(catnum[0] == fcat)]
    fs.loc[:,'prop'][i] = fs.loc[:,'frequency'][i]/cattot

# Make square
df_square = fs.pivot_table(index='first', columns='second', values='prop')
# Fill NA
df_square = df_square.fillna(0)
for i in range(0, df_square.shape[0]):
    for j in range(0, df_square.shape[0]):
        if i == j:
            df_square.iloc[i,j] = np.nan
# Plot
plt.figure(figsize=(30, 20))
sns.heatmap(df_square, annot=False, cmap='viridis', cbar=False, square=True)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60);
plt.savefig(kerasconf + '/keras_first_second_corr.png', bbox_inches='tight', dpi=300)
plt.clf


####### UP TO HERE
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_tile,
    facet_grid,
    labs,
    guide_legend,
    guides,
    theme,
    element_text,
    element_line,
    element_rect,
    theme_set,
    theme_void,
    element_blank
)
# Order groups according to category
temp = ktop3[["first", "second"]]
conv = adata.obs[["label__machine", "category"]]
conv = conv.reset_index()
conv = conv[["label__machine", "category"]]
conv = conv.drop_duplicates()
conv = conv.sort_values(by='category')
lm_order = np.array(conv.label__machine)
# Create a ggplot2-like heatmap using plotnine
fs = fs.sort_values(by='first')
plot_order = fs["first"]
fs['first'] = pd.Categorical(fs['first'], categories=lm_order)
fs['second'] = pd.Categorical(fs['second'], categories=lm_order)
heatmap = (ggplot(fs) 
           + aes(x='second', y='first', fill='prop')
           + geom_tile()
           + theme(axis_text_x=element_text(rotation=45, hjust=1, size=13))  # Rotate x-axis labels
           + theme(axis_text_y=element_text(size=13))
           + theme(figure_size=(20, 20))
           + theme(axis_title_x=element_text(size=20),  # Increase x-axis title size
                   axis_title_y=element_text(size=20)))
# Save the heatmap to a file
heatmap.save(kerasconf +  "/prop_second_calls_from_each_first.png", verbose=True)
plt.clf()













sns.clustermap(df_square, 
                   method = 'complete', 
                   annot  = False, 
                   annot_kws = {'size': 4})
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60);
plt.figure(figsize=(30, 20))
sns.clustermap(df_square, 
                   method = 'complete', 
                   annot  = False, 
                   annot_kws = {'size': 4})
plt.savefig(kerasconf + '/keras_first_second_corr.png', bbox_inches='tight', dpi=300)
plt.clf


# Sankey diagram - top vs second top hit per label
# Can we analyse this within category? How many times are cells called outside their category? 
temp = ktop3[["first", "second"]]
conv = adata.obs[["label__machine", "category"]]
conv = conv.reset_index()
conv = conv[["label__machine", "category"]]
conv = conv.drop_duplicates()



import plotly.graph_objects as go
import urllib, json
from pySankey.sankey import sankey

ktop3['first'] = ktop3['first'].astype("str")
sankey(
    ktop3['first'], ktop3['second'], aspect=20,
    fontsize=12, figure_name=kerasconf + '/sankey_first_second'
)



df = pd.read_csv(
    '/nfs/users/nfs_b/bh18/customer-goods.csv', sep=','
)
sankey(
    left=df['customer'], right=df['good'], rightWeight=df['revenue'], aspect=20,
    fontsize=20, figure_name="customer-good"
)











# First summarise the transfer from one to another
fs = ktop3.groupby(['first', 'second']).size().reset_index(name='frequency')
fs['second'] = remove_char(fs['second'] ,"probability__")
sankey(
    left=fs['first'], right=fs['second'], rightWeight=fs['frequency'], aspect=20,
    fontsize=20
)
fig = plt.gcf()
# Set size in inches
fig.set_size_inches(12, 12)
# Set the color of the background to white
fig.set_facecolor("w")
# Save the figure
fig.savefig(kerasconf + '/sankey_first_second/png', bbox_inches="tight", dpi=400)






