##### Bradley May 2024 ###
# Get the list of samples in the vcf file. So that we can adjust their labels in the expression data to match

# conda bcf
# vcf copied from: /lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/imputed_genotyped_march_2024/
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg
repo_dir=/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/bin/prep_for_qtl

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
bcftools query -l $vcf > samples_vcf.txt

# Find intersect with expression
expr_samps=/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/tables/genos_from_expression.txt
vcf_filt=CCF_OTAR-plates_12345_imputed_allchr-no_rsid_expr_filt.vcf.gz
bcftools view -S $expr_samps -Oz -o $vcf_filt $vcf --force-samples

# Compute PCA on these genotypes (after intersection)
mkdir -p plink_genotypes
plink2 --vcf $vcf_filt 'dosage=DS' --make-pgen --allow-extra-chr 0 --chr 1-22 XY --output-chr chrM --snps-only --rm-dup exclude-all --hwe 0.0000001 --out plink_genotypes/plink_genotypes
plink2 --freq counts --pfile plink_genotypes/plink_genotypes --out tmp_gt_plink_freq
plink2 --pca 5 --read-freq tmp_gt_plink_freq.acount  --pfile  plink_genotypes/plink_genotypes --out gtpca_plink

# Tidy
mkdir -p imputed_pca
for filename in *; do
    # Check if the file does not start with 'CCF' and does not contain 'sample' in its name
    if [[ ! $filename =~ ^CCF ]] && [[ ! $filename =~ sample ]]; then
        # Move the file to the destination directory
        mv "$filename" imputed_pca/
        echo "Moved: $filename"
    fi
done



# Plot the genotype PCs
Rscript ${repo_dir}/plot_geno_pcs_001.r

# Now plot a PCA using non-imputed data
mkdir -p pca && cd pca
# get non_imput vcf
no_imput_vcf=merged_plates_postqc_lifted_hg38_nomonom_RefAlt.vcf
cp /lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/${no_imput_vcf}  ./
vcf_filt=CCF_OTAR-plates_12345_NOT_imputed_allchr-no_rsid_expr_filt.vcf.gz # name of output file
awk {'print $1'} "../old_new_sample_IDs.txt" > old_id_expr_samps.txt
bcftools view -S old_id_expr_samps.txt -Oz -o $vcf_filt $no_imput_vcf --force-samples # Filter for old genotype IDs for samples with expression
mkdir -p plink_genotypes
plink2 --vcf $vcf_filt 'dosage=DS' --make-pgen --allow-extra-chr 0 --chr 1-22 XY --output-chr chrM --snps-only --rm-dup exclude-all --hwe 0.0000001 --out plink_genotypes/plink_genotypes
plink2 --freq counts --pfile plink_genotypes/plink_genotypes --out tmp_gt_plink_freq
plink2 --pca 5 --read-freq tmp_gt_plink_freq.acount  --pfile  plink_genotypes/plink_genotypes --out gtpca_plink

# Plot these
Rscript ${repo_dir}/plot_geno_pcs_002.r

# Make GRM using vcftools
module load HGI/softpack/users/bh18/bcftools/1.0
vcftools --gzvcf ${vcf_filt} --relatedness2

# Make and plot grm (wgcna conda)
Rscript ${repo_dir}/plot_grm.r

# Take europeans from 1kg, compute PCA for these, project our data into it

