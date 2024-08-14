##### Bradley May 2024 ###
# Get the list of samples in the vcf file. So that we can adjust their labels in the expression data to match

# conda bcf
# vcf copied from: /lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/imputed_genotyped_march_2024/
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/genotypes/Mar2024

vcf_files=$(find . -type f -name "*.dose.vcf.gz")
# index
for vcf in $vcf_files; do
    echo $vcf
    bcftools index $vcf
done
# merge
bcftools concat -Oz -o combined.vcf.gz $vcf_files




vcf=/lustre/scratch126/humgen/projects/sc-eqtl-ibd/data/genotypes/imputed_genotyped_march_2024/merged_chr.vcf.gz
bcftools query -l $vcf > samples_vcf.txt
