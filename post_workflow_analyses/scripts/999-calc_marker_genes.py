#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-08-12'
__version__ = '0.0.1'

# Load in the libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import argparse
import os
from matplotlib import rcParams
import plotnine as plt9

def most_and_second_most_frequent(series):
        counts = series.value_counts()
        most_common = counts.index[0]
        most_common_pct = counts.iloc[0] / counts.sum() * 100
        if len(counts) > 1:
            second_most_common = counts.index[1]
            second_most_common_pct = counts.iloc[1] / counts.sum() * 100
            return f"{most_common} ({most_common_pct:.1f}%), {second_most_common} ({second_most_common_pct:.1f}%)"
        else:
            return f"{most_common} ({most_common_pct:.1f}%)"
        
# 
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Calculation of marker genes
                """
        )
    
    parser.add_argument(
            '-i', '--input_file',
            action='store',
            dest='input_file',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-tissue', '--tissue',
            action='store',
            dest='tissue',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-s', '--subset',
            action='store',
            dest='subset',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-sv', '--subset_value',
            action='store',
            dest='subset_value',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-cc', '--cluster_column',
            action='store',
            dest='cluster_column',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-m', '--method',
            action='store',
            dest='method',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-f', '--filter',
            action='store',
            dest='filter',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-migf', '--min_in_group_fraction',
            action='store',
            dest='min_in_group_fraction',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-mogf', '--max_out_group_fraction',
            action='store',
            dest='max_out_group_fraction',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-mfc', '--min_fold_change',
            action='store',
            dest='min_fold_change',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-ca', '--compare_annots',
            action='store',
            dest='compare_annots',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-em', '--external_marker_sets',
            action='store',
            dest='em',
            required=False,
            help=''
        )
    
    parser.add_argument(
            '-b', '--baseout',
            action='store',
            dest='baseout',
            required=True,
            help=''
        )
    
    return parser.parse_args()


def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    tissue = inherited_options.tissue
    subset = inherited_options.subset
    subset_value = inherited_options.subset_value
    cluster_column = inherited_options.cluster_column
    method = inherited_options.method
    filter = inherited_options.filter
    min_in_group_fraction = inherited_options.min_in_group_fraction
    max_out_group_fraction = inherited_options.max_out_group_fraction
    min_fold_change = inherited_options.min_fold_change
    compare_annots = inherited_options.compare_annots
    compare_annots = compare_annots.split(",")
    external_marker_sets = inherited_options.external_marker_sets
    external_marker_sets = external_marker_sets.split(",")
    baseout = inherited_options.baseout
    
    # Testing:
    # input_file = "results_round2/all_Mesenchymal/tables/clustering_array/leiden_1.0/adata_PCAd_batched_umap_1.0.h5ad"
    # tissue = "combined"
    # subset = "manual_lineage"
    # subset_value = "Mesenchymal"
    # cluster_column = "leiden"
    # method = "wilcoxon"
    # filter = "True"
    # min_in_group_fraction = 0.5
    # max_out_group_fraction = 0.3
    # min_fold_change = 1.5
    # baseout = "results_round2"
    # compare_annots = ["Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels","Azimuth:predicted.celltype.l1", "Azimuth:predicted.celltype.l2", "Azimuth:predicted.celltype.l3", "Celltypist:cluster-ti-cd_healthy-freeze005_clustered_final:predicted_labels"]
    # external_markers = "external_markers/helmsley_23.txt"
    
    # Load in the input file
    print("~~~~~~~~~ Loading object ~~~~~~~~~")
    adata = sc.read_h5ad(input_file)
    print(f"adata shape: {adata.shape}")
    
    # Subset if desired
    if subset is not None:
        print("~~~~~~~~~ Subsetting ~~~~~~~~~")
        adata = adata[adata.obs[subset] == subset_value]
    
    ############## 1. Marker gene analysis ###########
    # Calculate markers
    print("~~~~~~~~~ Calculating marker genes ~~~~~~~~~")
    sc.tl.rank_genes_groups(adata, layer="log1p_cp10k", groupby=cluster_column, method=method)
    
    # Filter if desired
    if filter is not None:
        print("~~~~~~~~~ Filtering marker genes ~~~~~~~~~")
        sc.tl.filter_rank_genes_groups(adata, min_in_group_fraction=float(min_in_group_fraction), max_out_group_fraction=float(max_out_group_fraction), min_fold_change=float(min_fold_change))
        key="rank_genes_groups_filtered" 
    else:
        key="rank_genes_groups"
    
    # Define pathout
    outdir = f"{baseout}/{tissue}/markers/{subset}_{subset_value}_{cluster_column}_{method}"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
        
    # Save a dataframe of the results (use the unfiltered always here)
    marker_genes = adata.uns["rank_genes_groups"]
    group_temp = []
    for group in np.unique(adata.obs[cluster_column]):
        group_temp.append(pd.DataFrame({"gene_names": marker_genes['names'][group], "logfoldchanges": marker_genes['logfoldchanges'][group], "pvals": marker_genes['pvals'][group], "pvals_adj": marker_genes["pvals_adj"][group], cluster_column: group}))
    
    marker_all = pd.concat(group_temp)
    marker_all.dropna(subset=['gene_names'], inplace=True)
    conv = adata.var[["gene_symbols"]].reset_index()
    conv.rename(columns={"index": "gene_names"}, inplace =True)
    marker_all = marker_all.merge(conv, on="gene_names", how="left")
    print("~~~~~~~~~ Saving matrix ~~~~~~~~~")
    marker_all.to_csv(f"{outdir}/marker_sumstats.txt.gz", compression="zip", index=False, sep = "\t")
    
    # Plot results
    sc.settings.figdir=outdir
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
    
    # Panel
    rcParams['figure.figsize'] = 4,4
    rcParams['axes.grid'] = True
    print("Plotting panel")
    sc.pl.rank_genes_groups(adata, save="_panel_unfiltered.png", key="rank_genes_groups", gene_symbols="gene_symbols")
    
    # Dotplot (filtered and unfiltered)
    print("~~~~~~~ Plotting dotplot ~~~~~~~~~")
    sc.pl.rank_genes_groups_dotplot(adata, save="_top4.png", key="rank_genes_groups", n_genes=5, gene_symbols="gene_symbols")
    if filter:
        print("~~~~~~~~~ & filtered version ~~~~~~~~~")
        sc.pl.rank_genes_groups_dotplot(adata, layer = "log1_cp10k", save="_filtered_top4.png", key=key, n_genes=5, gene_symbols="gene_symbols")
    
    ############## 2. Comparing leiden against other annotations ###########
    # Summarise the top hits per gene
    # Group by 'leiden' and apply the custom function to each annot_col
    result = adata.obs.groupby('leiden')[compare_annots].agg(lambda x: most_and_second_most_frequent(x))
    result.to_csv(f"{outdir}/top_votes_other_annotations.csv")

    # Plot dotplots for each
    threshold=0.5 # Threshold for comparisons
    conf_score_cols = [x.replace("predicted_labels", "conf_score") for x in compare_annots]
    conf_score_cols = [x + ".score" if x.startswith("Azimuth") else x for x in conf_score_cols]
    df_cells_all = adata.obs[["leiden"] + compare_annots + conf_score_cols]
    for i, annot in enumerate(compare_annots):
        print(f"Calculating concordance between leiden and {annot}")
        # Sometimes get erros if predicted as na. Remove these
        df_cells = df_cells_all[~df_cells_all[conf_score_cols[i]].isna()]
        df_cells[annot] = df_cells[annot].astype('category')
        df1 = df_cells.groupby(["leiden", annot]).size().reset_index(name='nr_cells_label')
        df2 = df_cells.groupby(["leiden"]).size().reset_index(name='nr_cells_cluster')
        df_plt = pd.merge(df1, df2, on="leiden")
        df_plt['frac_cells_label'] = (df_plt['nr_cells_label'] / df_plt['nr_cells_cluster'])
        #df_cells[annot] = df_cells[annot].astype(str)
        df_cells[conf_score_cols[i]] = df_cells[conf_score_cols[i]].astype(float)
        temp = []
        for l in np.unique(df_cells['leiden']):
            for g in np.unique(df_cells[annot]):
                dat = df_cells.loc[(df_cells[annot] == g) & (df_cells["leiden"] == l), conf_score_cols[i]]
                val = dat.sum()/len(dat)
                to_add = pd.DataFrame({"leiden":[l], annot:[g], conf_score_cols[i]: val})
                temp.append(to_add)
        df3 = pd.concat(temp)
        #df3 = df_cells.groupby(["leiden", annot]).mean(conf_score_cols[i]).reset_index()
        #df3.columns.values[1] = conf_score_cols[i]
        df_plt = pd.merge(df_plt, df3, on=["leiden", annot])
        df_plt['probability_threshold']=df_plt[conf_score_cols[i]] > threshold
        df_plt[annot]=df_plt[annot].astype('str')
        df_plt["leiden"]=df_plt["leiden"].astype('str')
        cell_types=df_plt[annot].drop_duplicates()
        assignment=df_plt[["leiden", annot, 'frac_cells_label']].sort_values('frac_cells_label', ascending=False)
        assignment_all=assignment.drop_duplicates(["leiden"])
        assignment_all['label_forall']=assignment_all[annot]
        mask = assignment_all[annot].duplicated(keep=False)
        assignment_all.loc[mask, 'label_forall'] += " (" + assignment_all.groupby(annot).cumcount().add(1).astype(str) + ")"
        assignment_all["leiden"]=assignment_all["leiden"].astype('str')
        assignment_all[annot]=assignment_all[annot].astype('str')
        assignment_all['label_forall']=assignment_all['label_forall'].astype('str')
        assignment_highfrac=assignment_all.drop_duplicates(subset=annot,keep='first')
        assignment_highfrac["leiden"]=pd.to_numeric(assignment_highfrac["leiden"])
        assignment_highfrac.sort_values(by=["leiden"], inplace=True)
        assignment_highfrac["leiden"]=assignment_highfrac["leiden"].astype('str')
        cell_types_notassigned=cell_types[~cell_types.isin(assignment_highfrac[annot].values)].values
        clusters=df_plt["leiden"].drop_duplicates()
        clusters_notassigned=clusters[~clusters.isin(assignment_highfrac["leiden"].values)].values
        assignment_all["leiden"]=pd.to_numeric(assignment_all["leiden"])
        assignment_all.sort_values(by=["leiden"], inplace=True)
        assignment_all["leiden"]=assignment_all["leiden"].astype('str')
        assignment_all_drop=assignment_all.drop_duplicates(subset=annot,keep='first')
        clusters_assigned=assignment_highfrac["leiden"].values
        cell_types_assigned=assignment_highfrac[annot].values
        cell_type_order=np.concatenate([cell_types_assigned, cell_types_notassigned], axis=0)
        cluster_order=np.concatenate([clusters_assigned, clusters_notassigned], axis=0)
        df_plt[annot]=pd.Categorical(df_plt[annot], categories=cell_type_order)
        df_plt["leiden"]=pd.Categorical(df_plt["leiden"], categories=cluster_order)
        # Plot
        gplt = plt9.ggplot(df_plt, plt9.aes(
                x="leiden",
                y=annot
            ))
        gplt = gplt + plt9.theme_bw()
        gplt = gplt + plt9.theme(axis_text_x=plt9.element_text(angle=90))
        gplta = gplt + plt9.geom_point(
                plt9.aes(
                    color=conf_score_cols[i],
                    size='frac_cells_label')) # alpha='nr_cells_Keras:predicted_celltype')
        gplta = gplta + plt9.labs(
                title='',
                x='',
                y='',
                color='Predicted probability',
                size='Fraction of auto-annot cell type in cluster')
        gplta = gplta + plt9.theme(figure_size=(8, 8)) + plt9.ggtitle(f"Leiden vs {annot}")
        gplta.save(
                f'{outdir}/leiden_vs_{annot}.png',
                width=12,
                height=9
        )
    
    ############## 2. Comparing against Helmsley gut markers ###########
    for e in external_marker_sets:
        print(f"Comparing markers to {e}")
        external = pd.read_csv(external_markers, sep = "\t")
        external = external.loc[:, ~external.columns.str.contains('^Unnamed')]
        external = external[~external['Lineage'].isna()]
        external = external[~external['Celltype markers'].isna()]
        external_dict = dict()
        for index, row in external.iterrows():
            label = row['Cell label (abbreviation) - suggestion']
            markers = row['Celltype markers'].split(', ')  # Split the markers by ', ' and create a list
            markers = np.intersect1d(markers, adata.var['gene_symbols'])
            if len(markers) > 0:
                external_dict[label] = markers
        # Plot these
        sc.pl.dotplot(adata, external_dict, layer = "log1p_cp10k", groupby='leiden', gene_symbols="gene_symbols", save="_Helmsley_marker.png")


    

# Execute
if __name__ == '__main__':
    main()
    
