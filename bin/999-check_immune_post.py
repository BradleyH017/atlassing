# Bradley Aug 2024
# Checking immune cells post clustering
from anndata.experimental import read_elem
from h5py import File
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
tissues = ["Myeloid", "B", "T"]
res = ["0.75", "1.0", "0.5"]


obs_list = []
for i, t in enumerate(tissues):
    f=f"results/all_{t}/tables/clustering_array/leiden_{res[i]}/adata_PCAd_batched_umap_{res[i]}.h5ad"
    print(t)
    f2 = File(f, 'r')
    obs_list.append(read_elem(f2['obs']))


# Look at the number of healthy cells per tissue
for i,l in enumerate(obs_list):
    print(tissues[i])
    nhealthy = sum(l['disease_status'] == "Healthy")
    total=obs_list[i].shape[0]
    prophealthy = nhealthy/total
    perchealthy=prophealthy*100
    print(f"nHealthy = {nhealthy} ({perchealthy:.2f}%)")
    
    
# plot the vector of the proportions of healthy cells in each cluster, per lineage
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
for i, t in enumerate(obs_list):
    print(tissues[i])
    prop_healthy = t.groupby('leiden')['disease_status'].apply(lambda x: (x == 'Healthy').sum()/len(x)).values
    median = np.median(prop_healthy)
    plt.hist(prop_healthy, bins=20, edgecolor='black')

plt.xlabel('Proportion of each cluster comprised of healthy cells')
plt.axvline(x = 0.5, color = 'black', linestyle = '--', alpha = 0.5)
plt.legend()
ax.set(xlim=(0, 1))
plt.savefig(f"../results/prop_healthy_per_immune_cluster.png", bbox_inches='tight')
plt.clf()

# Plot per lineage against nCells
cols=["lightblue", "orange", "lightgreen"]
plt.figure(figsize=(8, 6))
for i, t in enumerate(obs_list):
    print(tissues[i])
    prop_healthy = t.groupby('leiden')['disease_status'].apply(lambda x: (x == 'Healthy').sum()/len(x))
    ncells = t['leiden'].value_counts()
    tog = pd.DataFrame({"nCells": ncells.values, "Prop_healthy": prop_healthy.values})
    tog['log_ncells'] = np.log10(tog['nCells'])
    median = np.median(tog['Prop_healthy'])
    plt.scatter(tog['Prop_healthy'],
                tog['log_ncells'],
                s=100, color=cols[i], label=f"{tissues[i]}. Median: {median:.2f}")
    
    for i, txt in enumerate(tog.index.astype(str)):
        plt.text(tog['Prop_healthy'][i], tog['log_ncells'][i], str(txt), ha='center', va='center')

plt.axvline(x = 0.5, color = 'black', linestyle = '--', alpha = 0.5)
plt.legend()
plt.xlabel('Proportion from healthy samples')
plt.ylabel('log10(nCells)')
plt.savefig(f"../results/nCells_vs_prop_healthy_immmune.png", bbox_inches='tight')
plt.clf()
