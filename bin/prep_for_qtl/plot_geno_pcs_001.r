# Bradley August 2024
# wgcna conda

# Load packages
library(dplyr)
library(ggrepel)
library(ggplot2)

# Specify dir
resdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg"

# Load PCs
pcs = read.delim(paste0(resdir, "/imputed_pca/gtpca_plink.eigenvec"))
colnames(pcs)[1] = "Genotyping_ID"

# Plot PCA coloured labelling sample ID
pc_comps = list(c("PC1", "PC2"), c("PC1", "PC3"), c("PC2", "PC3"))

for(i in seq_along(pc_comps)){
  print(pc_comps[[i]])
  x=pc_comps[[i]][1]
  y=pc_comps[[i]][2]
  p=ggplot(pcs, aes_string(x = x, y = y)) +
    geom_point(size = 3) +  
    geom_text(aes(label = Genotyping_ID), vjust = -1, hjust = 0.5) + 
    theme_bw() +
    labs(title = "Genotype PCs",
        x = x,
        y = y)

  ggsave(paste0(resdir, "/gt_pc", x, "_", y, ".png"), p, width = 6.5, height = 5, dpi = 300)
}

# genos in 1kg
kg_proj=read.delim("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/1000g_projectedpc.txt", sep = " ") # load projection of our data into 1kg
map = read.delim("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/plate_1-4_mapping.txt", header=F, sep = " ") # load sample ID conversion
colnames(map) = c("IID", "Genotyping_ID") #382 samples
kg_proj = left_join(kg_proj, map, by = "IID") # merge 

found = unique(kg_proj$Genotyping_ID)
found = found[!is.na(found)] # 369 genos converted
kg_proj$in_vcf = kg_proj$Genotyping_ID %in% pcs$Genotyping_ID
sum(kg_proj$in_vcf) # 312 are in the vcf

found_in_vcf = intersect(found, pcs$Genotyping_ID)
missing = setdiff(pcs$Genotyping_ID, found_in_vcf) #84 are to be found

# Of the missing set, how many of these can we grep and find? (Making sure they are not the 1KG set)
kg_proj$refset = grepl("^(NA|HG)", kg_proj$IID) # Add 1KG set
grep_missing = kg_proj[grep(paste(missing, collapse="|"), kg_proj$IID),]
grep_missing = grep_missing[grep_missing$refset != T,] # Make sure not 1kg set
nrow(grep_missing) # Rescued 84 / 84
# Add labels for the ones we can grep
for(m in missing){
  row = kg_proj[grep(m, kg_proj$IID),]
  if(nrow(row) == 1){
    kg_proj$Genotyping_ID[which(kg_proj$IID == row$IID)] = m
  }
}
# How many do we have now that are mapped and in vcf? 
pgfound = unique(kg_proj$Genotyping_ID)
pgfound = pgfound[!is.na(pgfound)]
# Subset for those in the vcf (post update)
pgfound_invcf = intersect(pgfound, pcs$Genotyping_ID) 
length(pgfound_invcf)
length(pgfound_invcf) == nrow(pcs) # all found
kg_proj$in_vcf = kg_proj$Genotyping_ID %in% pcs$Genotyping_ID # update this
# Plot
for(i in seq_along(pc_comps)){
  print(pc_comps[[i]])
  x=pc_comps[[i]][1]
  y=pc_comps[[i]][2]
  p <- ggplot(kg_proj, aes_string(x = x, y = y)) +
  geom_point(aes(color = ifelse(in_vcf, "yes", "no")), size = 1) + 
  #geom_text_repel(aes(label = label), color = "black", size = 3, 
  #                box.padding = 0.35, point.padding = 0.5, max.overlaps = Inf) + 
  scale_color_manual(values = c("yes" = "orange", "no" = "grey")) + 
  theme_bw() +
  labs(title = "Genotypes in 1KG PCs",
       x = x,
       y = y,
       color = "In Expression-subset vcf")
    
  ggsave(paste0(resdir, "/expr_gt_1kg_", x, "_", y, ".png"), p, width = 6.5, height = 5, dpi = 300)
}


# Also merge with pop ref to plot this
popref = read.delim("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/1000g_projected_popref.txt", sep = " ")
kg_popref = merge(kg_proj, popref, by="IID")
for(i in seq_along(pc_comps)){
  print(pc_comps[[i]])
  x=pc_comps[[i]][1]
  y=pc_comps[[i]][2]
  p <- ggplot(kg_popref, aes_string(x = x, y = y, color="Population")) +
    geom_point(size = 1) + 
    theme_bw() +
    labs(title = "1KG PCs",
        x = x,
        y = y,
        color = "Population reference") 

  ggsave(paste0(resdir, "/1kg_pc", x, "_", y, "_popref.png"), p, width = 6.5, height = 5, dpi = 300)
}

# Also save a conversion between all the old sample names and the one we now have
conv = kg_proj[kg_proj$in_vcf, c("IID", "Genotyping_ID")]
write.table(conv, "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg/old_new_sample_IDs.txt", col.names=F, quote=F, row.names=F, sep = "\t")
