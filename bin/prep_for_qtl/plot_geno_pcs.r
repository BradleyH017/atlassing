# Bradley August 2024
# wgcna conda

# Load packages
library(dplyr)
library(ggrepel)
library(ggplot2)

# Specify dir
resdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg"

# Load PCs
pcs = read.delim(paste0(resdir, "/gtpca_plink.eigenvec"))
colnames(pcs)[1] = "Genotyping_ID"

# Plot PCA coloured labelling sample ID
p=ggplot(pcs, aes(x = PC1, y = PC2)) +
  geom_point(size = 3) +  
  geom_text(aes(label = Genotyping_ID), vjust = -1, hjust = 0.5) + 
  theme_bw() +
  labs(title = "Genotype PCs",
       x = "PC1",
       y = "PC2")

ggsave(paste0(resdir, "/gt_pc1_2.png"), p, width = 6.5, height = 5, dpi = 300)

# genos in 1kg
kg_proj=read.delim("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/1000g_projectedpc.txt", sep = " ")
map = read.delim("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/plate_1-4_mapping.txt", header=F, sep = " ")
colnames(map) = c("IID", "Genotyping_ID")
kg_proj = left_join(kg_proj, map, by = "IID")
kg_proj$Genotyping_ID = ifelse(is.na(kg_proj$Genotyping_ID), kg_proj$IID, kg_proj$Genotyping_ID)
kg_proj$is_in_expr_set <- ifelse(kg_proj$Genotyping_ID %in% pcs$Genotyping_ID | !grepl("^(NA|HG)", kg_proj$Genotyping_ID),"yes","no") # Make sure we keep all those that are not 1kG in there, even if not the right label
kg_proj$label = ifelse(kg_proj$Genotyping_ID %in% c("SC00270", "FR36838035", "3981576836866"), kg_proj$Genotyping_ID, "") 
kg_proj$label = ifelse(grepl("^(SC00270|FR36838035|3981576836866)", kg_proj$IID), kg_proj$IID, kg_proj$label) # Rescue other two not mapped
kg_proj$label = ifelse(kg_proj$PC2 < 0.01 & kg_proj$is_in_expr_set == "yes", kg_proj$IID, kg_proj$label) # Also strange ones
kg_proj$label = ifelse(kg_proj$label %in% map$IID, kg_proj$Genotyping_ID, kg_proj$label )

p <- ggplot(kg_proj, aes(x = PC1, y = PC2)) +
  geom_point(aes(color = ifelse(is_in_expr_set == "yes", "yes", "no")), size = 1) + 
  geom_text_repel(aes(label = label), color = "black", size = 3, 
                  box.padding = 0.35, point.padding = 0.5, max.overlaps = Inf) + 
  scale_color_manual(values = c("yes" = "orange", "no" = "grey")) + 
  theme_bw() +
  labs(title = "Genotypes in 1KG PCs",
       x = "PC1",
       y = "PC2",
       color = "In Expression Set")  # Update legend title for clarity
    
ggsave(paste0(resdir, "/expr_gt_1kg_pc1_2.png"), p, width = 6.5, height = 5, dpi = 300)

# Also merge with pop ref to plot this
popref = read.delim("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/1000g_projected_popref.txt", sep = " ")
kg_popref = merge(kg_proj, popref, by="IID")
p <- ggplot(kg_popref, aes(x = PC1, y = PC2, color=Population)) +
  geom_point(size = 1) + 
  theme_bw() +
  labs(title = "1KG PCs",
       x = "PC1",
       y = "PC2",
       color = "Population reference") 

ggsave(paste0(resdir, "/1kg_pc1_2_popref.png"), p, width = 6.5, height = 5, dpi = 300)


