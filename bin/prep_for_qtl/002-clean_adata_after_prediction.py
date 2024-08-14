#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-05-20'
__version__ = '0.0.1'

import scanpy as sc
import pandas as pd 
import numpy as np


# Define adata (complete, annotated), columns to threshold and filter
h5 = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results/combined/objects/keras_full_annot_not_subset.h5ad"
cols_to_threshold = ["predicted_celltype_probability", "round1__predicted_celltype_probability"]
threshold = 0.5

# Load in the anndata object
adata = sc.read_h5ad(h5)

# Filter 
print(f"Original adata shape: {adata.shape}")
for c in cols_to_threshold:
    adata = adata[adata.obs[c] > threshold]
    print(f"After filter for {c}, adata shape: {adata.shape}")


# Adjust the genotype column so that it matches the vcf in as many cases as possible (got these from prep_for_qtl/001)
# To see the vcf sample identifiers, look at this:
vcf_samples="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Mar2024/samples_vcf.txt"
vcf_samps = pd.read_table(vcf_samples, header=None)
# expr_samps = np.unique(adata.obs['Post_QC_Genotyping_ID'].astype(str))

# Merge pre-QC genotype ID from the gut meta based on patient_id
gut_meta = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Mar2024/gut_meta.csv")
pre_pat = gut_meta[["patient_id", "Pre_QC_Genotyping_ID"]]
pre_pat.drop_duplicates(inplace=True)
pre_pat = pre_pat.dropna(subset=['Pre_QC_Genotyping_ID'])
adata.obs.reset_index(inplace=True)
pre_pat = pre_pat[pre_pat['patient_id'].isin(adata.obs['patient_id'])]
pre_pat = pre_pat.drop_duplicates(subset=['patient_id'])
temp = adata.obs.merge(pre_pat, on="patient_id", how="left")
keep = temp['cell']
adata = adata[adata.obs['cell'].isin(keep)] # Losing some cells that have missing patient_id ?!
adata.obs.reset_index("cell", inplace=True)
adata.obs = adata.obs.merge(pre_pat, on="patient_id", how="left")
adata.obs.set_index("cell", inplace=True)
adata.obs['patient_id'] = adata.obs['patient_id'].astype(str)
adata.obs['Pre_QC_Genotyping_ID'] = adata.obs['Pre_QC_Genotyping_ID'].astype(str)

# If post gt ID is empty, use pre
adata.obs['Post_QC_Genotyping_ID'] = adata.obs['Post_QC_Genotyping_ID'].astype(str)
adata.obs['Post_QC_Genotyping_ID'].replace('nan', np.nan, inplace=True)
adata.obs['Post_QC_Genotyping_ID'] = adata.obs['Post_QC_Genotyping_ID'].fillna(adata.obs['Pre_QC_Genotyping_ID'])


# If sample looks like "39815768577", replace this with the standard form, e.g 3.9815768577e+012
orig = adata.shape[0]
def convert_to_scientific(val):
    if str(val).startswith('39'):
        # Convert to scientific notation with 10 decimal places
        scientific_str = f"{float(val):.10e}"
        # Split the scientific notation to separate the exponent
        base, exponent = scientific_str.split('e')
        # Format the exponent to ensure it has three digits
        exponent = int(exponent)  # Convert exponent to int to remove leading zeros if any
        formatted_exponent = f"{exponent:+04d}"  # Ensure exponent has + and 3 digits, using +04d for formatting
        return f"{base}e{formatted_exponent}"
    return val

print(f"After filtration, have {adata.shape[0]} cells, from input of {orig}")
# Apply the function to the column
adata.obs['adjusted_PostQC_genotype_ID'] = adata.obs['Post_QC_Genotyping_ID'].apply(convert_to_scientific)
# Add the jargon
#adata.obs['adjusted_PostQC_genotype_ID'] = adata.obs['adjusted_PostQC_genotype_ID'].astype(str) + ".CEL_" + adata.obs['adjusted_PostQC_genotype_ID'].astype(str) + ".CEL"

# Find the missing sample IDs by converting them all to the patient IDs
gut_meta = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Mar2024/gut_meta.csv")
missing = np.setdiff1d(vcf_samps, adata.obs['adjusted_PostQC_genotype_ID'])
def remove_jargon(s):
    # Split the string by "_"
    first_part = s.split("_")[0]
    # Remove the ".CEL" suffix
    return first_part.replace(".CEL", "")

#adata.obs['no_jargon_id'] = adata.obs['adjusted_PostQC_genotype_ID'].apply(remove_jargon)
vcf_gt_ids = np.unique(vcf_samps)
adata.obs['adjusted_PostQC_genotype_ID'] = adata.obs['adjusted_PostQC_genotype_ID'].astype(str)
for v in vcf_gt_ids:
    # Get the proper name
    v = v.split("_")[0]
    v = v.replace(".CEL", "")
    # Find the corresponding patient ID for this sample from the anndata
    if v in adata.obs['patient_id'].values:
        print(f"Fixed {v}")
        adata.obs.loc[adata.obs['patient_id'] == v, "adjusted_PostQC_genotype_ID"] = v

# Now add the jargon
adata.obs['adjusted_PostQC_genotype_ID'] = adata.obs['adjusted_PostQC_genotype_ID'].astype(str) + ".CEL_" + adata.obs['adjusted_PostQC_genotype_ID'].astype(str) + ".CEL"
found = np.intersect1d(adata.obs['adjusted_PostQC_genotype_ID'], vcf_gt_ids)
missing = np.setdiff1d(vcf_gt_ids, adata.obs['adjusted_PostQC_genotype_ID'])

# Use clusters to also add category and lineage from this annotation
# For now, these are taking the majority proportion of cells from each cluster in the original category machine, grouped by the manual lineage defined earlier
prelim_annots = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/temp/prelim_annot.csv")
adata.obs.rename(columns={"predicted_celltype": "label__machine_bh"}, inplace=True)
prelim_annots.rename(columns={"label__machine": "label__machine_bh", "category__machine": "category__machine_bh", "lineage__machine": "lineage__machine_bh"}, inplace=True)
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(prelim_annots, on="label__machine_bh", how="left")
adata.obs.set_index("cell", inplace=True)

# Add the annotation levels, e.g: "label__machine,category__machine,everything__machine"
# Will use predicted__celltype, category__machine, lineage__machine and everything__machine, but make sure these also contain tissue
adata.obs['everything__machine_bh'] = "Everything"
annot_cols = ["everything__machine_bh", "label__machine_bh", "lineage__machine_bh", "category__machine_bh"]
for a in annot_cols:
    print(a)
    adata.obs[f"{a}_tissue"] = adata.obs[a].astype(str) + "_" + adata.obs['tissue'].astype(str)

# intersect witht the samples with genotype
vcf_sample_f="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Mar2024/samples_vcf.txt"
vcf_samps = pd.read_table(vcf_sample_f, header=None)
adata = adata[adata.obs["adjusted_PostQC_genotype_ID"].isin(vcf_samps[0].values)]
print(f"After intersection with genotypes, have {adata.shape[0]} cells, from input of {len(np.unique(adata.obs['samp_tissue']))} samples and {len(np.unique(adata.obs['patient_id']))} indviduals")

# Save this 
adata.write_h5ad("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results/combined/objects/keras_full_annot_subset_0pt5.h5ad")

# TEMP
# Save per aggregation column per group per tissue
aggregation_columns = ["everything__machine_bh_tissue","label__machine_bh_tissue","category__machine_bh_tissue"]
pathOut = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/old/v7_pre_tissue_swap/results/combined/objects"
for agg in aggregation_columns:
    levels = np.unique(adata.obs[agg])
    print(f"For {agg}, there are {len(levels)} groups")

# Divide by tissue
tissues = np.unique(adata.obs['tissue'])
for t in tissues:
    print(t)
    adata[adata.obs['tissue'] == t].write_h5ad(f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/old/v7_pre_tissue_swap/results/combined/objects/keras_full_annot_{t}_subset_0pt5.h5ad")