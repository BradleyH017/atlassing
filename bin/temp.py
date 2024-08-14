import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
lins = np.unique(inputobs['manual_lineage'])
for column in cols:
    plt.figure(figsize=(6, 4))
    fig,ax = plt.subplots(figsize=(8,6))
    for l in lins:
        data = inputobs.loc[inputobs["manual_lineage"] == l,column].values
        sns.distplot(data, hist=False, rug=True, label=l)
    #
    plt.legend()
    plt.xlabel(column)
    plt.savefig(f"../all_per_lineage_{column}.png", bbox_inches='tight')
    plt.clf()

# Plot nCells
df = pd.DataFrame(inputobs['manual_lineage'].value_counts())
df['log_nCells'] = np.log10(df['count'])
df.reset_index(inplace=True)
colors = plt.cm.get_cmap('tab10')(np.arange(len(df['manual_lineage'])))
df = df.sort_values(by='manual_lineage')
plt.figure(figsize=(6, 4))
plt.bar(df['manual_lineage'], df['count'],color=colors)
plt.xlabel('Manual_lineage')
plt.ylabel('nCells (x10^6)')
plt.savefig(f"../results/ncells_per_lineage.png", bbox_inches='tight')


temp = inputobs[inputobs['manual_lineage'] == "B"]
tissues = np.unique(temp['tissue'])
plt.figure(figsize=(6, 4))
fig,ax = plt.subplots(figsize=(8,6))
for l in tissues:
    data = temp.loc[temp["tissue"] == l,"log_n_genes_by_counts"].values
    sns.distplot(data, hist=False, rug=True, label=l)


plt.legend()
plt.xlabel("log_n_genes_by_counts")
plt.savefig(f"../results/b_per_lineage_ngene.png", bbox_inches='tight')
plt.clf()

# plot per tissue thresholds
plt.figure(figsize=(6, 4))
fig,ax = plt.subplots(figsize=(8,6))
for l in tissues:
    data = temp.loc[temp["tissue"] == l,"log_n_genes_by_counts"].values
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    cutoff_low = np.median(data) - (float(3) * mad)
    cutoff_high = np.median(data) + (float(3) * mad)
    plot = sns.distplot(data, hist=False, rug=True, label=f'{l} - (+/-{float(3)} MAD): {cutoff_low:.2f}-{cutoff_high:.2f}')
    line_color = plot.get_lines()[-1].get_color()
    plt.axvline(x = cutoff_low, linestyle = '--', alpha = 0.5, color=line_color)
    plt.axvline(x = cutoff_high, linestyle = '--', alpha = 0.5, color=line_color)


plt.legend()
plt.xlabel("log_n_genes_by_counts")
plt.savefig(f"../results/b_per_lineage_ngene_per_tissue.png", bbox_inches='tight')
plt.clf()

# Take outer
plt.figure(figsize=(6, 4))
fig,ax = plt.subplots(figsize=(8,6))
upper = []
lower = []
for l in tissues:
    data = temp.loc[temp["tissue"] == l,"log_n_genes_by_counts"].values
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    cutoff_low = np.median(data) - (float(3) * mad)
    cutoff_high = np.median(data) + (float(3) * mad)
    upper.append(cutoff_high)
    lower.append(cutoff_low)
    plot = sns.distplot(data, hist=False, rug=True, label=f'{l} - (+/-{float(3)} MAD): {cutoff_low:.2f}-{cutoff_high:.2f}')
    line_color = plot.get_lines()[-1].get_color()

plt.legend()
plt.xlabel("log_n_genes_by_counts")
plt.axvline(x = min(lower), linestyle = '--', alpha = 0.5, color="black")
plt.axvline(x = max(upper), linestyle = '--', alpha = 0.5, color="black")
plt.savefig(f"../results/b_per_lineage_ngene_per_tissue_outer.png", bbox_inches='tight')
plt.clf()