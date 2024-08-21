# Define which round of the analysis this is - will alter config and original input file
configfile: "configs/config_re_embed_after_within_lineage.yaml" # round 3 (post within lineage cluster QC)
input0 = expand("results_round2/{tissue}/objects/adata_grouped_post_cluster_QC.h5ad", tissue=config["tissue"])

rule all:
    input:
        expand("results/{tissue}/objects/celltypist_prediction.h5ad", tissue=config["tissue"]), expand("results/{tissue}/tables/lineage_model/base-model_report.tsv.gz", tissue=config["tissue"]), expand("results/{tissue}/objects/adata_PCAd_batched_umap.h5ad", tissue=config["tissue"]),expand("results/{tissue}/tables/sample_list.txt", tissue=config["tissue"])        

# Define memory increment function
def increment_memory(base_memory):
    def mem(wildcards, attempt):
        return base_memory * (2 ** (attempt - 1))
    return mem

rule qc_raw:
    input:
        input0
    output:
        "results/{tissue}/objects/adata_PCAd.h5ad",
        "results/{tissue}/tables/knee.txt",
        "results/{tissue}/objects/adata_unfilt_log1p_cp10k.h5ad"
    params:
        discard_other_inflams=config["discard_other_inflams"], 
        all_blood_immune=config["all_blood_immune"],
        min_nUMI=config["min_nUMI"],
        relative_grouping=config["relative_grouping"],
        relative_nMAD_threshold=config["relative_nMAD_threshold"],
        nMad_directionality=config["nMad_directionality"],
        threshold_method=config["threshold_method"],
        min_nGene=config["min_nGene"],
        MT_thresh_gut=config["MT_thresh_gut"],
        MT_thresh_blood=config["MT_thresh_blood"],
        use_MT_thresh_blood_gut_immune=config["use_MT_thresh_blood_gut_immune"],
        min_median_nCount_per_samp_blood=config["min_median_nCount_per_samp_blood"],
        min_median_nCount_per_samp_gut=config["min_median_nCount_per_samp_gut"],
        min_median_nGene_per_samp_blood=config["min_median_nGene_per_samp_blood"],
        min_median_nGene_per_samp_gut=config["min_median_nGene_per_samp_gut"],
        max_ncells_per_sample=config["max_ncells_per_sample"],
        min_ncells_per_sample=config["min_ncells_per_sample"],
        use_abs_per_samp=config["use_abs_per_samp"],
        filt_cells_pre_samples=config["filt_cells_pre_samples"],
        plot_per_samp_qc=config["plot_per_samp_qc"],
        filt_blood_keras=config["filt_blood_keras"],
        n_variable_genes=config["n_variable_genes"],
        hvgs_within=config["hvgs_within"],
        remove_problem_genes=config["remove_problem_genes"],
        per_samp_relative_threshold=config["per_samp_relative_threshold"],
        sample_level_grouping=config["sample_level_grouping"],
        cols_sample_relative_filter=config["cols_sample_relative_filter"],
        pref_matrix=config["pref_matrix"],
        calc_hvgs_together=config["calc_hvgs_together"]
    resources:
        mem=1300000, # all = 850000
        queue='teramem', # all = teramem
        mem_mb=1300000,
        mem_mib=1300000,
        disk_mb=1300000,
        tmpdir="tmp",
        threads=8 # all = 8
    conda:
        "scvi-env"
    shell:
        r"""
        python bin/001-raw_QC_update.py \
        --input_file {input} \
        --tissue {wildcards.tissue} \
        --discard_other_inflams {params.discard_other_inflams} \
        --all_blood_immune {params.all_blood_immune} \
        --min_nUMI {params.min_nUMI} \
        --nMad_directionality {params.nMad_directionality} \
        --threshold_method {params.threshold_method} \
        --relative_grouping {params.relative_grouping} \
        --relative_nMAD_threshold {params.relative_nMAD_threshold} \
        --min_nGene {params.min_nGene} \
        --MT_thresh_gut {params.MT_thresh_gut} \
        --MT_thresh_blood {params.MT_thresh_blood} \
        --use_MT_thresh_blood_gut_immune {params.use_MT_thresh_blood_gut_immune} \
        --min_median_nCount_per_samp_blood {params.min_median_nCount_per_samp_blood} \
        --min_median_nCount_per_samp_gut {params.min_median_nCount_per_samp_gut} \
        --min_median_nGene_per_samp_blood {params.min_median_nGene_per_samp_blood} \
        --min_median_nGene_per_samp_gut {params.min_median_nGene_per_samp_gut} \
        --max_ncells_per_sample {params.max_ncells_per_sample} \
        --min_ncells_per_sample {params.min_ncells_per_sample} \
        --use_abs_per_samp {params.use_abs_per_samp} \
        --filt_cells_pre_samples {params.filt_cells_pre_samples} \
        --plot_per_samp_qc {params.plot_per_samp_qc} \
        --filt_blood_keras {params.filt_blood_keras} \
        --n_variable_genes {params.n_variable_genes} \
        --hvgs_within {params.hvgs_within} \
        --remove_problem_genes {params.remove_problem_genes} \
        --per_samp_relative_threshold {params.per_samp_relative_threshold} \
        --sample_level_grouping {params.sample_level_grouping} \
        --cols_sample_relative_filter {params.cols_sample_relative_filter} \
        --pref_matrix {params.pref_matrix} \
        --calc_hvgs_together {params.calc_hvgs_together}
        """

# NOTE: Using the output of the above rule will mean using only HVGs
rule make_lineage_prediction_model:
    input:
        "results/{tissue}/objects/adata_PCAd.h5ad"
    output:
        "results/{tissue}/tables/lineage_model/base-model_report.tsv.gz"
    params:
        #combined_file=config["combined_file"],
        activation=config["activation"],
        loss=config["loss"],
        optimizer=config["optimizer"],
        sparsity_l1__activity=config["sparsity_l1__activity"],
        sparsity_l1__bias=config["sparsity_l1__bias"],
        sparsity_l1__kernel=config["sparsity_l1__kernel"],
        sparsity_l2__activity=config["sparsity_l2__activity"],
        sparsity_l2__bias=config["sparsity_l2__bias"],
        sparsity_l2__kernel=config["sparsity_l2__kernel"]
    resources:
        mem=850000,
        queue='teramem',
        mem_mb=850000,
        mem_mib=850000,
        disk_mb=850000,
        tmpdir="tmp",
        threads=4
    conda:
        "single_cell"
    shell:
        r"""
        path="results/combined/tables/keras-grid_search"
        mkdir -p ${{path}}

        echo -e "param__activation\t{params.activation}\nparam__loss\t{params.loss}\nparam__optimizer\t{params.optimizer}\nparam__sparsity_l1__activity\t{params.sparsity_l1__activity}\nparam__sparsity_l1__bias\t{params.sparsity_l1__bias}\nparam__sparsity_l1__kernel\t{params.sparsity_l1__kernel}\nparam__sparsity_l2__activity\t{params.sparsity_l2__activity}\nparam__sparsity_l2__bias\t{params.sparsity_l2__bias}\nparam__sparsity_l2__kernel\t{params.sparsity_l2__kernel}" > ${{path}}/keras-use_params.txt
        
        echo {input}

        python bin/005a-scanpy_cluster_optimise_model-keras_adata.py \
            --h5_anndata {input} \
            --leiden_res 0 \
            --number_epoch 25 \
            --keras_param_file ${{path}}/keras-use_params.txt \
            --batch_size 32 \
            --cluster_col "manual_lineage" \
            --train_size_fraction 0.67 \
            --output_file results/combined/tables/lineage_model/base
        """

# Get the batch methods to use
batch_methods = list(config["batch_correction_methods"])

if "scVI" in batch_methods:
    rule scVI:
        input:
            "results/{tissue}/objects/adata_PCAd.h5ad"
        output:
            "results/{tissue}/tables/batch_correction/scVI_matrix.npz"
        conda:
            "scvi-env_reserve"
        params:
            batch_column=config["batch_column"],
            correct_variable_only=config["correct_variable_only"]
        resources:
            mem=480000, # All = 300000
            queue='gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"
            mem_mb=480000,
            mem_mib=480000,
            disk_mb=480000,
            tmpdir="tmp",
            threads=8 # all = 8
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/tables/batch_correction

            python bin/001b-batch_correction.py \
                --tissue {wildcards.tissue} \
                --input_file {input[0]} \
                --batch_correction scVI \
                --batch_column {params.batch_column} \
                --correct_variable_only {params.correct_variable_only}
            
            """

if "scVI_default" in batch_methods:
    rule scVI_default:
        input:
            "results/{tissue}/objects/adata_PCAd.h5ad"
        output:
            "results/{tissue}/tables/batch_correction/scVI_default_matrix.npz"
        conda:
            "scvi-env_reserve"
        params:
            batch_column=config["batch_column"],
            correct_variable_only=config["correct_variable_only"]
        resources:
            mem=480000, # All = 300000
            queue='gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"
            mem_mb=480000,
            mem_mib=480000,
            disk_mb=480000,
            tmpdir="tmp",
            threads=8 # all = 8
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/tables/batch_correction

            python bin/001b-batch_correction.py \
                --tissue {wildcards.tissue} \
                --input_file {input[0]} \
                --batch_correction scVI_default \
                --batch_column {params.batch_column} \
                --correct_variable_only {params.correct_variable_only}
            
            """

if "Harmony" in batch_methods:
    rule Harmony:
        input:
            "results/{tissue}/objects/adata_PCAd.h5ad"
        output:
            "results/{tissue}/tables/batch_correction/Harmony_matrix.npz"
        conda:
            "scvi-env_reserve"
        params:
            batch_column=config["batch_column"],
            correct_variable_only=config["correct_variable_only"]
        resources:
            mem=300000, # All = 300000
            queue='long', # All = long
            mem_mb=300000,
            mem_mib=300000,
            disk_mb=300000,
            tmpdir="tmp",
            threads=8 # all = 8
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/tables/batch_correction

            python bin/001b-batch_correction.py \
                --tissue {wildcards.tissue} \
                --input_file {input[0]} \
                --batch_correction Harmony \
                --batch_column {params.batch_column} \
                --correct_variable_only {params.correct_variable_only}
            """

if "scANVI" in batch_methods:
    rule scANVI:
        input:
            "results/{tissue}/objects/adata_PCAd.h5ad"
        output:
            "results/{tissue}/tables/batch_correction/scANVI_matrix.npz"
        params:
            scANVI_col=config["scANVI_annot_column"],
            batch_column=config["batch_column"],
            correct_variable_only=config["correct_variable_only"]
        conda:
            "scvi-env_reserve"
        resources:
            mem=300000, # All = 300000
            queue='gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"
            mem_mb=300000,
            mem_mib=300000,
            disk_mb=300000,
            tmpdir="tmp",
            threads=8 # all = 8
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/tables/batch_correction

            python bin/001b-batch_correction.py \
                --tissue {wildcards.tissue} \
                --input_file {input[0]} \
                --scANVI_col {params.scANVI_col} \
                --batch_correction scANVI \
                --batch_column {params.batch_column} \
                --correct_variable_only {params.correct_variable_only}
            """

# Either benchmark batch correction and make a dummy file for the best result or make a dummy one of these from the choice in the config and gather anyway
# Gather input files
def gather_batches_within_tissue(wildcards):
    return expand("results/{tissue}/tables/batch_correction/{batch}_matrix.npz",
                  tissue=wildcards.tissue,
                  batch=config["batch_correction_methods"])

if config["benchmark_batch_correction"]:
    rule batch_correction_benchmarking:
        input:
            "results/{tissue}/objects/adata_PCAd.h5ad",
            gather_batches_within_tissue
        output:
            "results/{tissue}/tables/batch_correction/benchmark.csv",
            "results/{tissue}/tables/batch_correction/best_batch_method.txt",
            "results/{tissue}/objects/adata_PCAd_batched.h5ad",
            "results/{tissue}/tables/batch_correction/var_explained_pre_post_batch.csv"
        params:
            methods=config["batch_correction_methods"],
            label_integration=config["scANVI_annot_column"],
            batch_column=config["batch_column"],
            var_explained_cols=config["var_explained_cols"]
        conda:
            "scib-env_reserve"
        resources:
            mem=450000, # All = 350000
            queue='gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-normal -gpu - -m "farm22-gpu0203 farm22-gpu0204"
            mem_mb=450000,
            mem_mib=450000,
            disk_mb=450000,
            tmpdir="tmp",
            threads=8 # all = 8
        shell:
            r"""
            methods_combined="$(echo "{params.methods}" | sed 's/ /,/g')"
            echo $methods_combined

            python bin/001c-batch_correction_benchmarking.py \
                --input_file {input[0]} \
                --tissue {wildcards.tissue} \
                --methods $methods_combined \
                --use_label {params.label_integration} \
                --batch_column {params.batch_column} \
                --var_explained_cols {params.var_explained_cols}
            """
else:
    def get_batch_file(wildcards):
        return expand("results/{tissue}/tables/batch_correction/{method}_matrix.npz", 
                    tissue=wildcards.tissue, 
                    method=config["pref_matrix"])

    rule make_dummy_best_batch_file:
        input:
            "results/{tissue}/objects/adata_PCAd.h5ad",
            get_batch_file
        output:
            "results/{tissue}/tables/batch_correction/best_batch_method.txt",
            "results/{tissue}/objects/adata_PCAd_batched.h5ad"
        params:
            method=config["pref_matrix"]
        conda:
            "scvi-env_reserve"
        resources:
            mem=750000, # All =300000
            queue='teramem',
            mem_mb=750000,
            mem_mib=750000,
            disk_mb=750000,
            tmpdir="tmp",
            threads=2
        shell:
            r"""
            echo X_{params.method} >> {output[0]}

            python bin/001d-combine_adata_batch.py \
                --input_file {input[0]} \
                --tissue {wildcards.tissue} \
                --pref_matrix {params.method}
            """

if config["use_bbknn"]:
    rule get_nn_bbknn:
        input:
            "results/{tissue}/objects/adata_PCAd.h5ad",
            "results/{tissue}/tables/knee.txt"
        output:
            "results/{tissue}/tables/0_nn_bbknn.txt"
        params:
            use_matrix="X_pca"
        resources:
            mem=increment_memory(750000),
            queue='teramem',
            mem_mb=increment_memory(750000),
            mem_mib=increment_memory(750000),
            disk_mb=increment_memory(750000),
            tmpdir="tmp",
            threads=16
        conda:
            "bbknn"
        shell:
            r"""
            python bin/002-bbknn.py \
                --tissue {wildcards.tissue} \
                --fpath {input[0]} \
                --knee_file {input[1]} \
                --use_matrix {params.use_matrix}
            """

if "Manual" in config["nn_choice"]:
    rule make_dummy_nn_from_user:
        output:
            "results/{tissue}/tables/0_nn.txt"
        params:
            nn=config["nn"]
        resources:
            mem=5000,
            queue='long',
            mem_mb=5000,
            mem_mib=5000,
            disk_mb=5000,
            tmpdir="tmp",
            threads=1
        conda:
            "scvi-env_reserve"
        shell:
            r"""
            echo {params.nn} > {output}
            """

if "array" in config["nn_choice"]:
    rule nn_array:
        input:
            "results/{tissue}/objects/adata_PCAd_batched.h5ad",
            "results/{tissue}/tables/knee.txt",
            "results/{tissue}/tables/batch_correction/best_batch_method.txt"
        output:
            "results/{tissue}/tables/nn_array/{nn_value}_summary.csv",
            "results/{tissue}/tables/nn_array/adata_PCAd_batched_{nn_value}.h5ad"
        params:
            batch_column=config["batch_column"]
        conda:
            "scib-env"
        resources:
            mem=increment_memory(750000), # All = 350000
            queue='teramem', # All = long
            mem_mb=increment_memory(750000),
            mem_mib=increment_memory(750000),
            disk_mb=increment_memory(750000), 
            tmpdir="tmp",
            threads=32 # All =32
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/tables/nn_array

            python bin/002b-nn_array.py \
                --tissue {wildcards.tissue} \
                --input_file {input[0]} \
                --pref_matrix {input[2]} \
                --knee_file {input[1]} \
                --nn_val {wildcards.nn_value} \
                --batch_column {params.batch_column}
            """

    # Group the nn values and decide best
    def gather_nn_within_tissue(wildcards):
        return expand("results/{tissue}/tables/nn_array/{nn_value}_summary.csv",
                    tissue=wildcards.tissue,
                    nn_value=config["nn_values"])
    
    rule summarise_nn_array:
        input:
            gather_nn_within_tissue
        output:
            "results/{tissue}/tables/nn_array/summary_nn_array.txt",
            "results/{tissue}/tables/0_nn.txt"
        resources:
            mem=increment_memory(20000),
            queue='long',
            mem_mb=increment_memory(20000),
            mem_mib=increment_memory(20000),
            disk_mb=increment_memory(20000), 
            tmpdir="tmp",
            threads=2 
        shell:
            r"""
            cat {input} >> {output[0]}

            python bin/002c-summarise_nn_array.py \
                --summary_file {output[0]} \
                --tissue {wildcards.tissue}
            """


rule get_umap:
    input:
        "results/{tissue}/objects/adata_PCAd_batched.h5ad",
        "results/{tissue}/tables/knee.txt",
        "results/{tissue}/tables/0_nn.txt",
        "results/{tissue}/tables/batch_correction/best_batch_method.txt"
    output:
        "results/{tissue}/objects/adata_PCAd_batched_umap.h5ad"
    params:
        col_by=config["col_by"]
    resources:
        mem=increment_memory(750000), # All = 350000
        queue='teramem', # All = long
        mem_mb=increment_memory(750000),
        mem_mib=increment_memory(750000),
        disk_mb=increment_memory(750000), 
        tmpdir="tmp",
        threads=32 # All =32
    conda:
        "scvi-env_reserve"
    shell:
        r"""
        python bin/003-umap_embedding.py \
            --tissue {wildcards.tissue} \
            --fpath {input[0]} \
            --knee_file {input[1]} \
            --use_matrix {input[3]} \
            --optimum_nn_file {input[2]} \
            --col_by {params.col_by}
        """


rule make_dummy_keras_params:
    output:
        "results/{tissue}/tables/keras-grid_search/keras-use_params.txt"
    resources:
        mem=1000,
        queue='normal',
        mem_mb=1000,
        mem_mib=1000,
        disk_mb=1000,
        tmpdir="tmp",
        threads=1
    conda:
        "scvi-env_reserve"
    params:
        activation=config["activation"],
        loss=config["loss"],
        optimizer=config["optimizer"],
        sparsity_l1__activity=config["sparsity_l1__activity"],
        sparsity_l1__bias=config["sparsity_l1__bias"],
        sparsity_l1__kernel=config["sparsity_l1__kernel"],
        sparsity_l2__activity=config["sparsity_l2__activity"],
        sparsity_l2__bias=config["sparsity_l2__bias"],
        sparsity_l2__kernel=config["sparsity_l2__kernel"]
    shell:
        r"""
        mkdir -p results/{wildcards.tissue}/tables/keras-grid_search

        echo -e "param__activation\t{params.activation}\nparam__loss\t{params.loss}\nparam__optimizer\t{params.optimizer}\nparam__sparsity_l1__activity\t{params.sparsity_l1__activity}\nparam__sparsity_l1__bias\t{params.sparsity_l1__bias}\nparam__sparsity_l1__kernel\t{params.sparsity_l1__kernel}\nparam__sparsity_l2__activity\t{params.sparsity_l2__activity}\nparam__sparsity_l2__bias\t{params.sparsity_l2__bias}\nparam__sparsity_l2__kernel\t{params.sparsity_l2__kernel}" > {output}
        """

rule test_clusters_make_model:
    input:
        "results/{tissue}/objects/adata_PCAd_batched_umap.h5ad", # Differs from other snakefiles
        "results/{tissue}/tables/batch_correction/best_batch_method.txt",
        "results/{tissue}/tables/keras-grid_search/keras-use_params.txt"
    output:
        "results/{tissue}/tables/clustering_array/leiden_0/base-model_report.tsv.gz",
        "results/{tissue}/tables/clustering_array/leiden_0/base-.weights.h5"
    resources:
        mem=increment_memory(1000000), #All - 850000
        queue='teramem',
        mem_mb=increment_memory(1000000),
        mem_mib=increment_memory(1000000),
        disk_mb=increment_memory(1000000),
        tmpdir="tmp",
        threads=16 # All 16
    conda:
        "single_cell"
    shell:
        r"""
        mkdir -p results/{wildcards.tissue}/tables/clustering_array
        mkdir -p results/{wildcards.tissue}/tables/clustering_array/leiden_0
        python bin/005a-scanpy_cluster_optimise_model-keras_adata.py \
            --h5_anndata {input[0]} \
            --leiden_res 0 \
            --number_epoch 25 \
            --keras_param_file {input[2]} \
            --batch_size 32 \
            --train_size_fraction 0.67 \
            --output_file results/{wildcards.tissue}/tables/clustering_array/leiden_0/base
        """

rule make_celltypist_model:
    input:
        input0
    output:
        "results/{tissue}/objects/leiden-adata_grouped_post_cluster_QC.pkl"
    resources:
        mem=1400000,
        queue='teramem',
        mem_mb=1400000,
        mem_mib=1400000,
        disk_mb=1400000,
        tmpdir="tmp",
        threads=40
    conda:
        "scvi-env"
    shell:
        r"""
        python bin/993-make_celltypist_model.py \
            --h5_anndata {input} \
            --outdir results/{wildcards.tissue}/objects \
            --annotation "leiden" \
            --gene_symbols
        """



rule assess_celltypist_model:
    input:
        input0,
        "results/{tissue}/objects/leiden-adata_grouped_post_cluster_QC.pkl"
    output:
        "results/{tissue}/objects/celltypist_prediction.h5ad"
    resources:
        mem=800000,
        queue='teramem',
        mem_mb=800000,
        mem_mib=800000,
        disk_mb=800000,
        tmpdir="tmp",
        threads=40
    conda:
        "scvi-env"
    shell:
        r"""
        python bin/994-assess_celltypist_autoannot.py \
            --h5_anndata {input[0]} \
            --outdir results/{wildcards.tissue}/objects \
            --model {input[1]}
        """


rule save_sample_list:
    output:
        "results/{tissue}/tables/sample_list.txt"
    conda:
        "scvi-env"
    resources:
            mem=300000,
            queue='normal',
            mem_mb=300000,
            mem_mib=300000,
            tmpdir="tmp",
            threads=2
    params:
        orig_h5ad=config["original_input_round1"],
        samp_col=config["batch_column"]
    shell:
        r"""
        python bin/008a-get_adata_sample_list.py \
                --h5ad {params.orig_h5ad} \
                --samp_col "sanger_sample_id" \
                --output_file "results/{wildcards.tissue}/tables/"
        """

# Run CellTypist manually
# mkdir -p results/combined/figures/annotation
# mkdir -p results/combined/tables/annotation/
# mkdir -p results/combined/figures/UMAP/annotation
# mkdir -p results/{tissue}/tables/annotation/CellTypist/
# python bin/007-CellTypist.py \
#            --tissue "combined" \
#            --h5_path "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results_round3/combined/objects/adata_PCAd_batched_umap.h5ad" \
#            --pref_matrix "results_round3/combined/tables/batch_correction/best_batch_method.txt" \
#            --model "/lustre/scratch127/cellgen/cellgeni/cakirb/celltypist_models/megagut_celltypist_lowerGI+lym_adult_mar24.pkl" \
#            --model_name "Megagut_adult_lower_GI_mar24" 