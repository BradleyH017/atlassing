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

# Derive category info
add_cat = adata.obs[["label", "category"]]
add_cat = add_cat.reset_index()
add_cat = add_cat[["label", "category"]]
add_cat = add_cat.drop_duplicates()

# Define paths
compfigpath = figpath + "/anno_conf"
kerasconf = compfigpath + "/keras"
ctconf = compfigpath + "/celltypist"
if os.path.exists(compfigpath) == False:
    os.mkdir(compfigpath)

if os.path.exists(kerasconf) == False:
    os.mkdir(kerasconf)

if os.path.exists(ctconf) == False:
    os.mkdir(ctconf)

# Plot the distribution of keras for all and for individual categories
cats_all = np.unique(adata.obs.category)
cats_all = np.append(cats_all, "All_cells")
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for c in cats_all:
    if c != "All_cells":
        idx = adata.obs.category == c
        dat = ktop3.first_confidence[idx]
        sns.distplot(dat, hist=False, rug=True, label=c)
    else:
        dat = ktop3.first_confidence
        sns.distplot(dat, hist=False, rug=True, label=c, color="black")

plt.legend()
ax.set(xlim=(0, 1))
plt.title(c)
plt.xlabel('Distribution of annotation confidence')
plt.savefig(kerasconf + '/keras_top_confidence_distribution.png', bbox_inches='tight')
plt.clf()



# Plot the proportion of the second versus the first annotation, within category and annotated by cell-type
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

# Identify the x-axis location of the most common confidence
labs = np.unique(adata.obs.label)
common_conf = pd.DataFrame({"label": labs, "ncells": 0, "common_conf": np.nan})
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for index,l in enumerate(labs):
    idx = adata.obs.label == l
    data = ktop3.first_confidence[idx] - ktop3.second_confidence[idx]
    common_conf.iloc[index,1] = data.shape[0]
    sns.distplot(data, hist=False, rug=True, color='black')
    sns.kdeplot(data, shade=True)
    ax = plt.gca()
    # Find the x-value corresponding to the peak
    try:
        kde_xdata = ax.get_lines()[index].get_xdata()
        peak_x = kde_xdata[ax.get_lines()[index].get_ydata().argmax()]
        common_conf.iloc[index,2] = peak_x
    except Exception as e:
        print(f"An error occurred for {l}: {e}")

plt.xlabel('1st - 2nd annotation confidence')
plt.savefig(kerasconf + '/all_keras_second_over_first.png', bbox_inches='tight')
plt.clf()

common_conf.to_csv(tabdir + '/keras_all_common_anno_conf.csv')

# Print those with poor annotation ( < 0.5)
common_conf[common_conf.common_conf < 0.5]

# Plot these, coloured within category and highlighting bad
common_conf['plot_label'] = common_conf.apply(lambda row: row['label'] if row['common_conf'] < 0.5 else None, axis=1)
subset_df = common_conf[common_conf.common_conf < 0.5]

plt.figure(figsize=(16, 12))
for c in cats:
    sns.scatterplot(data=common_conf[common_conf.category == c], x='ncells', y='common_conf', s=100, edgecolor='k')

for i, row in subset_df.iterrows():
    plt.text(row['ncells'], row['common_conf'], row['plot_label'],rotation=90, fontsize=10, ha='left', va='bottom', color='black')

plt.legend(cats)
plt.xlabel('Number of cells',fontsize=14)
plt.ylabel('Common confidence', fontsize=14)
plt.savefig(kerasconf + '/first_less_second_conf_vs_ncells.png', bbox_inches='tight')
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
# Limit this to when the confidence of the first annotation is < 0.5 and second is > 0.25
ktop3_int = ktop3[ktop3.first_confidence < 0.5]
ktop3_int = ktop3_int[ktop3_int.second_confidence > 0.25]
fs = ktop3_int.groupby(['first', 'second']).size().reset_index(name='frequency')
fs['prop'] = 0
catnum = np.unique(ktop3_int['first'], return_counts = True)
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
plt.setp(ax.set_xticklabels(), rotation=60);
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
temp = ktop3_int[["first", "second"]]
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

##############################################################
####################### CellTypist ###########################
##############################################################
from scipy import stats

# Load the Elmentaite annotations per cells probabilties from CellTypist
predictions = pd.read_csv(tabdir + "/CellTypist_Elmentaite_prob_matrix.csv", index_col=0)

# Merge onto the keras annotations to test against one another
keras_ct = keras.merge(predictions, left_index=True, right_index=True)

# Rename columns 
for column in keras_ct.columns:
    if not column.startswith('probability__'):
        keras_ct.rename(columns={column: 'Elm_' + column}, inplace=True)

keras_ct.columns = keras_ct.columns.str.replace('probability_', 'keras')

# Calculate the correlation between probailities of all cell type annotations from keras with those from Elmentaite
keras_columns = [col for col in keras_ct.columns if col.startswith('keras')]
elm_columns = [col for col in keras_ct.columns if col.startswith('Elm_')]

# Calculate the correlation
probcor = pd.DataFrame(index=keras_columns, columns=elm_columns)
# Calculate correlations and p-values
for keras_col in keras_columns:
    for elm_col in elm_columns:
        corr, p_value = stats.pearsonr(keras_ct[keras_col], keras_ct[elm_col])
        if p_value < 0.05 and np.abs(corr) > 0.5:
            probcor.loc[keras_col, elm_col] = corr
        else:
            probcor.loc[keras_col, elm_col] = 0

# Convert the correlation matrix to numeric values
probcor = probcor.apply(pd.to_numeric)

# Remove columns (CellTypist annotations) for which there is no signficiantly correlated keras probability
strongprobcor = probcor.iloc[:,(probcor.sum() > 0).values]
# Same for rows
strongprobcor = strongprobcor.iloc[(probcor.sum(axis=1) > 0).values,]


# Plot
plt.figure(figsize=(16, 12))
sns.clustermap(strongprobcor, annot=False, cmap='coolwarm', cbar=True,
               row_cluster=True, col_cluster=True, dendrogram_ratio=(.2, .2),
               cbar_kws={'label': 'Correlation'})
# Reduce axis label size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45)
plt.title('Probability correlation across all cells (p<0.05, abs(cor) > 0.5)')
plt.savefig(figpath + '/elm_ct_vs_keras_cormat.png', bbox_inches='tight', dpi=300)
plt.clf

# Quantify the proportional agreement between individual cell types from Elmentaite and those from keras (TI)
predictions['elm_best_guess_probability'] = predictions.max(axis=1)
predictions['elm_best_guess'] = predictions.idxmax(axis=1)
predictions['elm_best_guess'] = predictions.apply(lambda row: row['elm_best_guess'] if row[row['elm_best_guess']] > 0.5 else 'uncertain', axis=1)

# Add this to the adata object
adata.obs = adata.obs.merge(predictions[["elm_best_guess_probability", "elm_best_guess"]], left_index=True, right_index=True)
sc.pl.umap(adata, color = "elm_best_guess_probability", frameon=True, save="_post_batch_post_batch_cCellTypist_Elmentaite_best_guess_probability.png")
sc.pl.umap(adata, color = "elm_best_guess", frameon=True, save="_post_batch_post_batch_cCellTypist_Elmentaite_best_guess.png", palette = "Set1")

# Also plot majority voting
adata.obs['majority_voting'] = adata.obs['majority_voting'].astype('category')
sc.pl.umap(adata, color = "majority_voting", frameon=True, save="_post_batch_post_batch_CellTypist_Elmentaite_majority_voting.png", palette = "Set1")


# Merge this with the majority voting data
ct_labels = pd.read_csv(tabdir + "/CellTypist_labels.csv", index_col=0)
ct_labels = ct_labels.merge(predictions[["elm_best_guess", "elm_best_guess_probability"]], left_index=True, right_index=True)

# Plot the finer predictions on a UMAP
import matplotlib as mp
adata.obs = adata.obs.merge(ct_labels, left_index=True, right_index=True)
cats = np.unique(adata.obs.category)
for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    # If there is less than 20 cells of a given label, make this NA
    category_counts = temp.obs['predicted_labels'].value_counts()
    # Identify annotations with less than 1% of cells in that category 
    lim = temp.shape[0]*0.01
    categories_to_replace = category_counts[category_counts < lim].index
    # Replace those categories with 'NA'
    temp.obs['predicted_labels'] = temp.obs['predicted_labels'].apply(lambda x: 'NA' if x in categories_to_replace else x)
    print(c)
    sc.pl.umap(temp, color="predicted_labels", title = "keras category - " + c + " - Elmentaite CellTypist", frameon=True, save="_post_batch_post_" + c+ "_celltypist_elm_fine.png", palette = "Dark2")
    sc.pl.umap(temp, color="elm_best_guess_probability", title = "keras category - " + c + " - Elmentaite CellTypist confidence", frameon=True, save="_post_batch_post_" + c+ "_confidence_celltypist_elm_fine.png")

# Claulate the proportion of keras labels per CellTypist label
grouped = adata.obs.groupby(['label', 'predicted_labels']).size().reset_index(name='count')
pivot_table = pd.pivot_table(grouped, values='count', index='label', columns='predicted_labels', fill_value=0)
# Calculate proportions
proportion_matrix = pivot_table.div(pivot_table.sum(axis=1), axis=0)
# Summarise the max and max proportion in a new dataframe and save
prop_sum = pd.DataFrame({"Elm_best_guess": "", "proportion": 0},  index = proportion_matrix.index)
prop_sum["Elm_best_guess"] =  proportion_matrix.idxmax(axis=1)
prop_sum["proportion"] = proportion_matrix.max(axis=1)
prop_sum.to_csv(tabdir + "/Elmentaite_celltypist_proporion_summary_per_keras.csv")


# Look at the ditribution of first vs second probabilities for CellTypist Annotations (just like done with the keras)
filtered_columns = [col for col in predictions.columns if 'elm' not in col]
# Calculate the second greatest value for each row in the filtered columns
predictions['elm_next_best_guess'] = predictions[filtered_columns].apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
# Calculated the difference in these scores
predictions['elm_first_less_second'] = predictions['elm_best_guess_probability'] - predictions['elm_next_best_guess']
# Add keras category
keras_cat = adata.obs[["category"]]
predictions = predictions.merge(keras_cat, left_index=True, right_index=True)

# Plot
cats_all = np.unique(adata.obs.category)
cats_all = np.append(cats_all, "All_cells")
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for c in cats_all:
    if c != "All_cells":
        dat = predictions[predictions.category == c].elm_first_less_second
        sns.distplot(dat, hist=False, rug=True, label=c)
    else:
        dat = predictions.elm_first_less_second
        sns.distplot(dat, hist=False, rug=True, label=c, color="black")

plt.legend()
ax.set(xlim=(0, 1))
plt.title(c)
plt.xlabel('Distribution of First - Second annotation confidence')
plt.savefig(ctconf + '/CellTypist_first_less_second_confidence_distribution.png', bbox_inches='tight')
plt.clf()

# Now plot within category (Doesn't work well, because of the discordance between keras and CellTypist annotations, get a huge number of CellTypist predictions per keras category)
#cats = np.unique(adata.obs.category)
#for c in cats:
#    labels = np.unique(predictions[predictions.category == c].elm_best_guess)
#    plt.figure(figsize=(8, 6))
#    fig,ax = plt.subplots(figsize=(8,6))
#    for l in labels:
#        idx = (predictions.category == c) & (predictions.elm_best_guess == l)
#        dat = predictions.elm_first_less_second[idx]
#        sns.distplot(dat, hist=False, rug=True, label=l)
#    plt.legend()
#    ax.set(xlim=(0, 1))
#    plt.title(c)
#    plt.xlabel('1st - 2nd Elmentaite CellTypist annotation confidence')
#    plt.savefig(kerasconf + '/' + c + '_Elmentaite_CellTypist_second_over_first.png', bbox_inches='tight')
#    plt.clf()

# Find the peak distribution of first - second per celltypist annotation (Including the uncertain cells)
predictions['elm_best_guess_inc_uncertain'] = predictions[filtered_columns].idxmax(axis=1)
labs = np.unique(predictions.elm_best_guess_inc_uncertain)
common_conf = pd.DataFrame({"Elm_top_guess": labs, "ncells": 0, "common_conf": np.nan})
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for index,l in enumerate(labs):
    print(l)
    idx = predictions.elm_best_guess == l
    data = predictions.elm_first_less_second[idx]
    common_conf.iloc[index,1] = data.shape[0]
    # Find the x-value corresponding to the peak
    if data.shape[0] >= 5:
        sns.distplot(data, hist=False, rug=True, color='black')
        sns.kdeplot(data, shade=True)
        ax = plt.gca()
        kde_xdata = ax.get_lines()[len(ax.get_lines())-1].get_xdata()
        peak_x = kde_xdata[ax.get_lines()[len(ax.get_lines())-1].get_ydata().argmax()]
        common_conf.iloc[index,2] = peak_x
    else:
        print("Less than 5 cells, not calculated")

plt.xlabel('1st - 2nd annotation confidence')
plt.savefig(ctconf + '/all_CellTypist_second_over_first.png', bbox_inches='tight')
plt.clf()

common_conf.to_csv(ctconf + '/CellTypist_Elmentaite_all_common_anno_conf.csv')

# Print those with poor annotation ( < 0.5)
common_conf[common_conf.common_conf < 0.5]

# Plot these, coloured within category and highlighting bad
common_conf['plot_label'] = common_conf.apply(lambda row: row['Elm_top_guess'] if row['common_conf'] < 0.5 else None, axis=1)
subset_df = common_conf[common_conf.common_conf < 0.5]

plt.figure(figsize=(16, 12))
sns.scatterplot(data=common_conf, x='ncells', y='common_conf', s=100, edgecolor='k')
for i, row in subset_df.iterrows():
    plt.text(row['ncells'], row['common_conf'], row['plot_label'],rotation=25, fontsize=10, ha='left', va='bottom', color='black')

plt.xlabel('Number of cells',fontsize=14)
plt.ylabel('Common confidence', fontsize=14)
plt.savefig(ctconf + '/first_less_second_conf_vs_ncells.png', bbox_inches='tight')
plt.clf()


