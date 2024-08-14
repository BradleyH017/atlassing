####################################################
############ Plot the model test results ###########
####################################################

library(ggplot2)
library(ggsci)

# Load data
dir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/"
tissues=c("blood", "ti", "rectum", "gut", "all")
keras_res = lapply(tissues, function(x){
    fpath = paste0(dir, "/results/", x, "/tables/keras-grid_search/keras--grid_result.tsv.gz")
    temp = read.delim(gzfile(fpath))
    temp$tissue = x
    return(temp)
})
keras_res = do.call(rbind, keras_res)

# Make a column combining the model parameters for the hyperparam values
keras_res$gathered_vars = paste(keras_res$param_sparsity_l1__activity, keras_res$param_sparsity_l1__bias, keras_res$param_sparsity_l1__kernel, keras_res$param_sparsity_l2__activity, keras_res$param_sparsity_l2__bias, keras_res$param_sparsity_l2__kernel, sep = ",")

# For each tissue, get the best params and replace these values with tissue name
for(t in tissues){
  best_params = keras_res[keras_res$rank_test_score == 1 & keras_res$tissue == t,]$gathered_vars
  keras_res$gathered_vars = gsub(best_params, t, keras_res$gathered_vars)
}

# Also add this for best1/best2 and original
best_params <- list(
  best1 = "0.01,1e-04,1e-04,1e-04,0.01,0.01",
  best2 = "1e-04,1e-04,1e-04,1e-04,0.01,0.01",
  original = "0.01,0.01,0.01,0,0,0"
)


for (name in names(best_params)) {
  keras_res$gathered_vars[keras_res$gathered_vars == best_params[[name]]] <- name
}

# Subset for these
rows_to_remove = grep(",", keras_res$gathered_vars)
keras_res = keras_res[-rows_to_remove, ]

# Plot the mean_test_score 
keras_res = keras_res[, c("mean_test_score", "tissue", "gathered_vars")]

# Plot
min_score <- min(keras_res$mean_test_score) - 0.01
p <- ggplot(keras_res, aes(x = tissue, y = mean_test_score, fill = gathered_vars)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Tissue", y = "Mean Test Score (CV=5)") +
  theme_bw() + 
  scale_fill_npg(scale_name="gathered_vars") +  # Using the npg color scheme
  coord_cartesian(ylim = c(min_score, 1)) +  # Limit y-axis to minimum value - 0.01 to 1
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(face = "bold")) + 
  ggtitle("Optimum hyper-parameters across cell sets")

outdir=paste0(dir, "results/cross_tissue")
# Check if the directory exists
if (!file.exists(outdir)) {
  # Create the directory if it does not exist
  dir.create(outdir, recursive = TRUE)
}
ggsave(paste0(outdir, "/keras_models_across_tissues.png"), p, width = 6.5, height = 5, dpi = 300)

# Count number of times the two best models beat one another
count_greater_ti <- keras_res %>%
  filter(gathered_vars %in% c('ti', 'gut')) %>%
  group_by(tissue) %>%
  summarise(count = sum(mean_test_score[gathered_vars == 'ti'] > mean_test_score[gathered_vars == 'gut']))
