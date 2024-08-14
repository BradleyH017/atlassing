# Bradley August 2024
# Testing MRtree on myeloid cells
library(mrtree)
library(ggplot2)

# Set dir to load in from
dir = 'old/070824_MAD3_for_all/results/all_Myeloid/tables/clustering_array'
# Load in data
f = list.files(dir)
res = NULL
for(r in seq_along(f)){
    print(f[r])
    temp = read.csv(paste0(dir, "/", f[r], "/clusters.csv"))
    if (is.null(res)){
        res = temp
    } else {
        res = merge(res, temp, by="cell")
    }
}
rownames(res) = res$cell
res = res[,-which(colnames(res) == "cell")]

# Make mrtree
restree = mrtree(res)
saveRDS(restree "../results/myeloid_restree.Rds")