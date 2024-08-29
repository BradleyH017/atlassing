##### Bradley May 2024 ###
# Get the list of samples in the vcf file. So that we can adjust their labels in the expression data to match

# conda bcf
# vcf copied from: /lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/imputed_genotyped_march_2024/
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg

#vcf_files=$(find . -type f -name "*.dose.vcf.gz")
## index
#for vcf in $vcf_files; do
#    echo $vcf
#    bcftools index $vcf
#done
## merge
#bcftools concat -Oz -o combined.vcf.gz $vcf_files

# Get list of samples
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg
module load HGI/softpack/users/bh18/bcftools/1.0
vcf=CCF_OTAR-plates_12345_imputed_allchr-no_rsid.vcf.gz # imputed
vcf= # non-imputed - what we want for PCA
bcftools query -l $vcf > samples_vcf.txt

# Find intersect with expression
expr_samps=/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/tables/genos_from_expression.txt
vcf_filt=CCF_OTAR-plates_12345_imputed_allchr-no_rsid_expr_filt.vcf.gz
bcftools view -S $expr_samps -Oz -o $vcf_filt $vcf --force-samples

# Compute PCA on these genotypes (after intersection)
mkdir plink_genotypes
plink2 --vcf $vcf_filt 'dosage=DS' --make-pgen --allow-extra-chr 0 --chr 1-22 XY --output-chr chrM --snps-only --rm-dup exclude-all --hwe 0.0000001 --out plink_genotypes/plink_genotypes
plink2 --freq counts --pfile plink_genotypes/plink_genotypes --out tmp_gt_plink_freq
plink2 --pca 5 --read-freq tmp_gt_plink_freq.acount  --pfile  plink_genotypes/plink_genotypes --out gtpca_plink

# Plot the genotype PCs
