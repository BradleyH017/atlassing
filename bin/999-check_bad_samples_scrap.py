################## Bradley April 2024 #########################
#### Checking certain samples for nCells and nGenes pre QC ####

# Load packages
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt

# Load file
tissue="all"
fpath = f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined/input/adata_raw_input_{tissue}.h5ad"
adata = sc.read_h5ad(fpath)
adata.obs['samp_tissue'] = adata.obs['experiment_id'].astype('str') + "_" + adata.obs['tissue'].astype('str')

# Define bad samples
bad_samples = ["5892STDY11060804__donor_ti", "5892STDY13265991__donor_ti", "OTARscRNA12147503__donor_ti", "scrnacdb13098576__donor_blood", 
               "Crohns_Disease_Collection_Study8727394__donor_blood", "5892STDY8038978__donor_blood", "5892STDY10315938__donor_ti", 
               "5892STDY11395280__donor_ti", "OTARscRNA13669872__donor_ti", "OTARscRNA13166396__donor_ti", "5892STDY11060801__donor_ti", 
               "Crohns_Disease_Collection_Study8727200__donor_blood"]

# Define thresholds (same as config)
min_median_nCount_per_samp_blood = 0 
min_median_nCount_per_samp_gut = 0
min_median_nGene_per_samp_blood = 80 
min_median_nGene_per_samp_gut = 100


# Get stats per sample
depth_count = pd.DataFrame(index = np.unique(adata.obs.samp_tissue), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts", "Median_nCounts", "Median_nGene_by_counts"])
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,1] = adata.obs[adata.obs.samp_tissue == samp].shape[0]
    depth_count.iloc[s,0] = sum(adata.obs[adata.obs.samp_tissue == samp].total_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,3] = sum(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,4] = np.median(adata.obs[adata.obs.samp_tissue == samp].total_counts)
    depth_count.iloc[s,5] = np.median(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)
    if samp in bad_samples:
        depth_count.iloc[s,2] = "Red"
    else: 
        depth_count.iloc[s,2] = "Navy"

depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))
depth_count["log10_Median_nCounts"] = np.log10(np.array(depth_count["Median_nCounts"].values, dtype = "float"))

# Plot
tissues = np.unique(adata.obs['tissue'])
for t in tissues:
    samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count[depth_count.index.isin(samps)]["Median_nCounts"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
    plt.xlabel('Median counts / cell')
    plt.ylabel('Median genes detected / cell')
    if t == "blood":
        plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_blood}, min_median_nGene: {min_median_nGene_per_samp_blood}")
    else:
        plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_gut}, min_median_nGene: {min_median_nGene_per_samp_gut}")
    
    plt.savefig(f"temp/sample_median_counts_ngenes_{t}.png", bbox_inches='tight')
    plt.clf()
    
# What about on the basis of the number of cells? 
# Plot nCells vs Median_nGene_by_counts
for t in tissues:
    samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
    plt.xlabel('nCells')
    plt.ylabel('Median genes detected / cell')
    if t == "blood":
        plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nGene: {min_median_nGene_per_samp_blood}")
    else:
        plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nGene: {min_median_nGene_per_samp_gut}")
    
    plt.savefig(f"temp/sample_nCells_ngenes_{t}.png", bbox_inches='tight')
    plt.clf()
    
# Also seperate by disease status and add potential threshlds
adata = adata[adata.obs['disease_status'].isin(["cd", "healthy"])]
status = np.unique(adata.obs['disease_status'])
for t in tissues:
    for s in status:
        samps = adata.obs[(adata.obs['tissue'] == t) & (adata.obs['disease_status'] == s)]['samp_tissue']
        if(len(samps) > 0):
            print(f"{t} - {s}")
            plt.figure(figsize=(8, 6))
            plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
            plt.xlabel('nCells')
            plt.ylabel('Median genes detected / cell')
            if t == "blood":
                plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.title(f"{t}, {s} - min_median_nGene: {min_median_nGene_per_samp_blood}")
            else:
                plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.title(f"{t}, {s} - min_median_nGene: {min_median_nGene_per_samp_gut}")
            
            plt.savefig(f"temp/sample_nCells_ngenes_{t}_{s}.png", bbox_inches='tight')
            plt.clf()

# Plot with proposed cut offs, taking into account target cell numbers for each
target_cd = 6000
target_healthy = 3000        
# Add disease status
disease_meta = adata.obs[["samp_tissue", "disease_status"]].drop_duplicates()
depth_count.reset_index(inplace=True)
depth_count = depth_count.rename(columns={"index": "samp_tissue"})
depth_count = depth_count.merge(disease_meta, on="samp_tissue", how="left")
depth_count.set_index("samp_tissue", inplace=True)
for t in tissues:
    if t == "ti":
        for s in status:
            samps = adata.obs[(adata.obs['tissue'] == t) & (adata.obs['disease_status'] == s)]['samp_tissue']
            if(len(samps) > 0):
                print(f"{t} - {s}")
                plt.figure(figsize=(8, 6))
                plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
                plt.xlabel('nCells')
                plt.ylabel('Median genes detected / cell')
                if s == "healthy":
                    temp_orig = depth_count[(depth_count.index.isin(samps)) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)]
                    temp_new = temp_orig[(temp_orig['nCells'] < 3*target_healthy) & (temp_orig['nCells'] > target_healthy/3)]
                    loss = temp_orig.shape[0]-temp_new.shape[0]
                    print(f"nCell cut off leads to loss of {loss} samples")
                    plt.axvline(x = 3*target_healthy, color = 'red', linestyle = '--', alpha = 0.5)
                    plt.axvline(x = target_healthy/3, color = 'red', linestyle = '--', alpha = 0.5)
                    plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                    plt.title(f"{t}, {s} - min_median_nGene: {min_median_nGene_per_samp_gut}, nCells {target_healthy/3}-{3*target_healthy}")
                else:
                    temp_orig = depth_count[(depth_count.index.isin(samps)) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)]
                    temp_new = temp_orig[(temp_orig['nCells'] < 3*target_cd) & (temp_orig['nCells'] > target_cd/3)]
                    loss = temp_orig.shape[0]-temp_new.shape[0]
                    print(f"nCell cut off leads to loss of {loss} samples")
                    plt.axvline(x = 3*target_cd, color = 'red', linestyle = '--', alpha = 0.5)
                    plt.axvline(x = target_cd/3, color = 'red', linestyle = '--', alpha = 0.5)
                    plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                    plt.title(f"{t}, {s} - min_median_nGene: {min_median_nGene_per_samp_gut}, nCells - {target_cd/3}-{3*target_cd}")
                
                plt.savefig(f"temp/sample_nCells_ngenes_{t}_{s}_prop_cutoffs.png", bbox_inches='tight')
                plt.clf()
    
    if t == "blood":
        samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
        temp_orig = depth_count[(depth_count.index.isin(samps)) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)]
        temp_new = temp_orig[(temp_orig['nCells'] < 3*target_cd) & (temp_orig['nCells'] > target_cd/3)]
        loss = temp_orig.shape[0]-temp_new.shape[0]
        print(f"nCell cut off leads to loss of {loss} samples")
        plt.figure(figsize=(8, 6))
        plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
        plt.xlabel('nCells')
        plt.ylabel('Median genes detected / cell')
        plt.axvline(x = 3*target_cd, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = target_cd/3, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nGene: {min_median_nGene_per_samp_blood}, nCells - {target_cd/3}-{3*target_cd}")
        plt.savefig(f"temp/sample_nCells_ngenes_{t}_prop_cutoffs.png", bbox_inches='tight')
        plt.clf()
    
    if t == "rectum":
        samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
        temp_orig = depth_count[(depth_count.index.isin(samps)) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)]
        temp_new = temp_orig[(temp_orig['nCells'] < 3*target_healthy) & (temp_orig['nCells'] > target_healthy/3)]
        loss = temp_orig.shape[0]-temp_new.shape[0]
        print(f"nCell cut off leads to loss of {loss} samples")
        plt.figure(figsize=(8, 6))
        plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
        plt.xlabel('nCells')
        plt.ylabel('Median genes detected / cell')
        plt.axvline(x = 3*target_healthy, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = target_healthy/3, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nGene: {min_median_nGene_per_samp_gut}, nCells - {target_healthy/3}-{3*target_healthy}")
        plt.savefig(f"temp/sample_nCells_ngenes_{t}_prop_cutoffs.png", bbox_inches='tight')
        plt.clf()
        
        
# Check the problem samples in the within lineage analysis in the depth_counts
bad_samples_in_lineage = ["scrnacdb13098576__donor_blood", "5892STDY13265991__donor_ti", "OTARscRNA13669872__donor_ti", "5892STDY11060801__donor_ti"]
depth_count['in_lineage_bad_sample_col'] = ""
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    colindx = np.where(depth_count.columns == "in_lineage_bad_sample_col")[0]
    if samp in bad_samples_in_lineage:
        depth_count.loc[samp,"in_lineage_bad_sample_col"] = "Red"
    else: 
        depth_count.loc[samp,"in_lineage_bad_sample_col"] = "Navy"


for t in tissues:
    samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["in_lineage_bad_sample_col"], alpha=0.7)
    plt.xlabel('nCells')
    plt.ylabel('Median genes detected / cell')
    if t == "blood":
        plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nGene: {min_median_nGene_per_samp_blood}")
    else:
        plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nGene: {min_median_nGene_per_samp_gut}")
    
    plt.savefig(f"temp/sample_nCells_ngenes_{t}_bad_samps_from_within_lineage.png", bbox_inches='tight')
    plt.clf()