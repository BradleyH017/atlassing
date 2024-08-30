# Bradley August 2024
library(reshape2)
library(ComplexHeatmap)
library(circlize)
library(cluster)
library(dplyr)
library(tidyr)
library(tidyverse)
set.seed(123)

####### 1.  Plot relatedness
dir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg/pca"
relatedness = read.table(paste0(dir, "/out.relatedness2"), header=T)

# Make square
samps <- sort(unique(unlist(relatedness$INDV1)))
mat = matrix(nrow=length(samps), ncol=length(samps), relatedness$RELATEDNESS_PHI, dimnames = list(samps, samps))

# cluster
row_hc <- hclust(dist(mat, method = "euclidean"), method = "complete")
col_hc <- hclust(dist(t(mat), method = "euclidean"), method = "complete")

ppi=400
png(paste0(dir, "/relatedness_matrix_plot.png"), res=ppi, width=8*ppi, height=8*ppi)
Heatmap(as.matrix(mat),
        cluster_rows = row_hc,
        cluster_columns = col_hc,
        show_row_dend = TRUE,
        show_column_dend = TRUE,
        col = colorRamp2(c(0,0.5), c("white", "red")),
        name = "relatedness",
        column_title = "Sample 1 ",
        row_title = "Sample 2",
        heatmap_legend_param = list(title = "relatedness"),
        #row_dend_side = "right", column_dend_side = "bottom",
        row_names_gp = gpar(fontsize = 5),
        column_names_gp = gpar(fontsize = 5),  
        row_dend_width = unit(8, "mm"), 
        column_dend_height = unit(8, "mm"))

dev.off()

# check the vals themselves
nonself = relatedness[relatedness$INDV1 != relatedness$INDV2,]
nonself = nonself[order(-nonself$RELATEDNESS_PHI),]