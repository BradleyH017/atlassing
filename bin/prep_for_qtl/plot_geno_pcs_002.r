# Bradley August 2024
# wgcna conda

# Load packages
library(dplyr)
library(ggrepel)
library(ggplot2)

# Specify dir
resdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg/pca"

# Load PCs
pcs = read.delim(paste0(resdir, "/gtpca_plink.eigenvec"))
colnames(pcs)[1] = "IID"

# Load conversion
map = read.delim(paste0(resdir, "/../old_new_sample_IDs.txt"), sep = "\t", header=F)
colnames(map) = c("IID", "Genotyping_ID")
pcs = merge(pcs,map, by="IID")

# Plot PCA coloured labelling sample ID
pc_comps = list(c("PC1", "PC2"), c("PC1", "PC3"), c("PC2", "PC3"), c("PC3", "PC4"), c("PC4", "PC5"))

for(i in seq_along(pc_comps)){
  print(pc_comps[[i]])
  x=pc_comps[[i]][1]
  y=pc_comps[[i]][2]
  p=ggplot(pcs, aes_string(x = x, y = y)) +
    geom_point(size = 1) +  
    geom_text(aes(label = Genotyping_ID), vjust = -1, hjust = 0.5, size=3) + 
    theme_bw() +
    labs(title = "Genotype PCs",
        x = x,
        y = y)

  ggsave(paste0(resdir, "/gt_pc", x, "_", y, ".png"), p, width = 6.5, height = 5, dpi = 300)
}