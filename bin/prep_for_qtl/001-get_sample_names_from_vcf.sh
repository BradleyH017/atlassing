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

####### Take europeans from 1kg, compute PCA for these, project our data into it
# Define dirs
module load HGI/softpack/users/bh18/bcftools/1.0
popref=/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/march_2024/1000g_projected_popref.txt
workdir=/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Aug2024_ashg/1kg_b38_work
kg_dir=/lustre/scratch125/humgen/resources/1000g/release/20201028
mkdir -p $workdir

# Get europeans
grep "EUR" $popref | awk '{print $1}' > ${workdir}/european.txt

# check 1kg samp names
chr1_file=CCDG_14151_B01_GRM_WGS_2020-08-05_chr1.filtered.shapeit2-duohmm-phased.vcf.gz
bcftools query -l ${kg_dir}/${chr1_file} > ${workdir}/1kg_b38_samples.txt

# Subset samples from each chr
for c in {1..22}; do
    echo $c
    bcftools view -S ${workdir}/european.txt -Oz -o ${workdir}/1kg_chr${c}_eur.vcf.gz ${kg_dir}/CCDG_14151_B01_GRM_WGS_2020-08-05_chr${c}.filtered.shapeit2-duohmm-phased.vcf.gz --force-samples
    bcftools index ${workdir}/1kg_chr${c}_eur.vcf.gz
done

# Concatonate
bcftools concat ${workdir}/1kg_chr*_eur.vcf.gz -o ${workdir}/1kg_eur_merged.vcf.gz -Oz
mkdir -p ${workdir}/plink_genotypes

# Before making plink file, make sure we only use the intersection of variants from both datasets to compute embedding
# So find the variants on chromosome 1 in our data
vcf_filt=${workdir}/../pca/CCF_OTAR-plates_12345_NOT_imputed_allchr-no_rsid_expr_filt.vcf.gz # name of file containing our own genotypes, filtrered for those with expression
module load HGI/softpack/groups/team152/genetics_tools/1
tabix -p vcf $vcf_filt
mkdir -p ${workdir}/plink_genotypes_query
zcat $vcf_filt | grep -n "##" | tail -n 1  | awk '{print $1}' # 33 lines of header (+1 for columns) # Find how much header
zcat $vcf_filt | awk 'BEGIN {OFS="\t"} {if (NR > 34) $3=$1":"$2":"$4":"$5; print}' > ${workdir}/query_rename.vcf # rename var
bcftools query -f '%ID\n' ${workdir}/query_rename.vcf > ${workdir}/query_vars.txt # Vars from query
# Vars from ref
bcftools query -f '%ID\n' ${workdir}/1kg_eur_merged.vcf.gz > ${workdir}/ref_vars.txt 
# Intersection
sort ${workdir}/ref_vars.txt -o ${workdir}/ref_vars_sorted.txt
sort ${workdir}/query_vars.txt -o ${workdir}/query_vars_sorted.txt
comm -12 ${workdir}/query_vars_sorted.txt ${workdir}/ref_vars_sorted.txt  > ${workdir}/intersection_vars.txt
plink2 --vcf ${workdir}/1kg_eur_merged.vcf.gz 'dosage=DS' --extract ${workdir}/intersection_vars.txt --make-bed --out ${workdir}/plink_genotypes/plink_genotypes

# Get GRM, PCA and snp loadings using gcta
#module load HGI/softpack/groups/team152/wes-v1/1
#gcta64 --bfile ${workdir}/plink_genotypes/plink_genotypes --make-grm --out ${workdir}/plink_genotypes/grm
#gcta64 --bfile ${workdir}/plink_genotypes/plink_genotypes --pc-loading ${workdir}/gtpca_gcta --out gtpca_gcta # Get loadings
#gcta64 --grm ${workdir}/plink_genotypes/grm --pca 20  --out ${workdir}/gtpca_gcta

# OR using plink
plink2 --freq counts --bfile plink_genotypes/plink_genotypes --out ${workdir}/ref_gt_plink_freq
plink2 --pca 10 --read-freq ${workdir}/ref_gt_plink_freq.acount  --bfile  plink_genotypes/plink_genotypes --out gtpca_plink
plink2 --bfile plink_genotypes/plink_genotypes --freq counts --pca allele-wts --out ${workdir}/ref_gt_plink_freq

# Convert the input samples to bedfile, extracting these variants
mkdir -p ${workdir}/plink_genotypes_query
plink2 --vcf ${workdir}/query_rename.vcf 'dosage=DS' --extract ${workdir}/intersection_vars.txt --make-bed --out ${workdir}/plink_genotypes_query/plink_genotypes

# HERE

# Project using plink
plink2 --bfile ${workdir}/plink_genotypes_query/plink_genotypes \
    --read-freq ${workdir}/ref_gt_plink_freq.acount \
    --score ${workdir}/ref_gt_plink_freq.eigenvec.allele 2 5 \
    --score-col-nums 6-15 \
    --out query_in_ref_plink



# Using GCTA
#gcta64 --bfile ${workdir}/plink_genotypes_query/plink_genotypes --project-loading gtpca_gcta 20 --out TAR_pca20 > projection.out

# Get the 1kG subpop
cd ${workdir} && wget https://1000genomes.s3.us-east-1.amazonaws.com/release/20130502/integrated_call_samples_v3.20130502.ALL.panel

# plot 
Rscript 

