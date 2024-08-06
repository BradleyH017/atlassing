configfile: "config_cluster_within_lineage.yaml" # round2 config + cluster within lineage

rule all:
    input:
        expand("results/{tissue}/objects/adata_clusters_post_clusterQC.h5ad", tissue=config["tissue"])
        #expand("results/{tissue}/tables/summary_nn_array.txt", tissue=config["tissue"]), expand("results/{tissue}/tables/optimum_nn_bbknn.txt", tissue=config["tissue"])
        #expand("results/{tissue}/objects/adata_PCAd_batched_umap_clustered.h5ad", tissue=config["tissue"])
        #expand("results/{tissue}/objects/adata_clusters_post_clusterQC.h5ad", tissue=config["tissue"])
        #"results/combined/objects/adata_grouped_post_cluster_QC.h5ad"
        #"results/combined/objects/adata_grouped_post_cluster_QC.h5ad"
        #"results/combined/tables/annotation/CellTypist/CellTypist_prob_matrix.csv"
        #expand("results/{tissue}/tables/annotation/CellTypist/CellTypist_anno_conf.csv", tissue=config["tissue"])
        #expand("results/{tissue}/tables/annotation/CellTypist/CellTypist_anno_conf.csv", tissue=config["tissue"]), expand("results/{tissue}/objects/adata_raw_predicted_celltypes_filtered.h5ad", tissue=config["tissue"])
        

# Define memory increment function
def increment_memory(base_memory):
    def mem(wildcards, attempt):
        return base_memory * (2 ** (attempt - 1))
    return mem

rule qc_raw:
    input:
        "input_cluster_within_lineage/adata_manual_lineage_clean_{tissue}.h5ad"
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
        mem=lambda wildcards: 1100000 if wildcards.tissue == 'all_Epithelial' else 200000, 
        queue=lambda wildcards: 'teramem' if wildcards.tissue == 'all_Epithelial' else 'normal', 
        mem_mb=lambda wildcards: 1100000 if wildcards.tissue == 'all_Epithelial' else 200000,
        mem_mib=lambda wildcards: 1100000 if wildcards.tissue == 'all_Epithelial' else 200000,
        disk_mb=lambda wildcards: 1100000 if wildcards.tissue == 'all_Epithelial' else 200000,
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
            mem=250000, # All = 150000
            queue='gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"
            mem_mb=250000,
            mem_mib=250000,
            disk_mb=250000,
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
            mem=480000, # All = 150000
            queue='gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"
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
            mem=150000, # All = 150000
            queue='normal', # All = normal
            mem_mb=150000,
            mem_mib=150000,
            disk_mb=150000,
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
            mem=150000, # All = 150000
            queue='gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"
            mem_mb=150000,
            mem_mib=150000,
            disk_mb=150000,
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
            queue='gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"', # All = gpu-huge -gpu - -m "farm22-gpu0203 farm22-gpu0204"
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
            mem=100000, # All =150000
            queue='normal',
            mem_mb=100000,
            mem_mib=100000,
            disk_mb=100000,
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
            "results/{tissue}/tables/optimum_nn_bbknn.txt"
        params:
            use_matrix="X_pca"
        resources:
            mem=increment_memory(300000),
            queue='normal',
            mem_mb=increment_memory(300000),
            mem_mib=increment_memory(300000),
            disk_mb=increment_memory(300000),
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
            "results/{tissue}/tables/optimum_nn.txt"
        params:
            nn=config["nn"]
        resources:
            mem=5000,
            queue='normal',
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
            queue='normal', # All = normal
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
            "results/{tissue}/tables/optimum_nn.txt"
        resources:
            mem=increment_memory(20000),
            queue='normal',
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
        "results/{tissue}/tables/optimum_nn.txt",
        "results/{tissue}/tables/batch_correction/best_batch_method.txt"
    output:
        "results/{tissue}/objects/adata_PCAd_batched_umap.h5ad"
    params:
        col_by=config["col_by"]
    resources:
        mem=increment_memory(100000), # All = 350000
        queue='normal', # All = normal
        mem_mb=increment_memory(100000),
        mem_mib=increment_memory(100000),
        disk_mb=increment_memory(100000), 
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

# cluster the data with an array (save to individual anndata files)
rule cluster_array:
    input:
        "results/{tissue}/objects/adata_PCAd_batched_umap.h5ad",
        "results/{tissue}/tables/batch_correction/best_batch_method.txt"
    output:
        "results/{tissue}/tables/clustering_array/leiden_{clustering_resolution}/adata_PCAd_batched_umap_{clustering_resolution}.h5ad",
        "results/{tissue}/tables/clustering_array/leiden_{clustering_resolution}/clusters.csv",
        "results/{tissue}/tables/clustering_array/leiden_{clustering_resolution}/umap_clusters_res{clustering_resolution}.png"
    resources:
        mem=lambda wildcards: 60000 if wildcards.tissue == 'all_Epithelial' else 40000,
        queue='normal', # All = normal
        mem_mb=lambda wildcards: 60000 if wildcards.tissue == 'all_Epithelial' else 40000,
        mem_mib=lambda wildcards: 60000 if wildcards.tissue == 'all_Epithelial' else 40000,
        disk_mb=lambda wildcards: 60000 if wildcards.tissue == 'all_Epithelial' else 40000,
        tmpdir="tmp",
        threads=16 # All 16
    conda:
        "single_cell"
    shell:
        r"""
        mkdir -p results/{wildcards.tissue}/tables/clustering_array
        mkdir -p results/{wildcards.tissue}/tables/clustering_array/leiden_{wildcards.clustering_resolution}
        python bin/004-scanpy_cluster.py \
            --tissue {wildcards.tissue} \
            --fpath {input[0]} \
            --pref_matrix {input[1]} \
            --clustering_resolution {wildcards.clustering_resolution}
        """


if config["optimise_run_params"]:
    def get_mem_mb(wildcards, attempt):
        return attempt * 750000 # All 750000

    rule run_params_optimisation:
        input:
            "results/{tissue}/tables/clustering_array/leiden_3.0/clusters.csv",
            "results/{tissue}/tables/log1p_cp10k_sparse.npz",
            "results/{tissue}/tables/genes.txt",
            "results/{tissue}/tables/cells.txt"
        output:
            "results/{tissue}/tables/keras-grid_search/keras-use_params.txt" 
        params:
            optimise_res=config["keras_optimisation_res"]
        resources:
            mem=get_mem_mb,
            queue='normal', # All = normal
            mem_mb=get_mem_mb,
            mem_mib=get_mem_mb,
            disk_mb=get_mem_mb,
            tmpdir="tmp",
            threads=20 # Aids parallelisation. For all , use 20 threads and 750000 memory 
        conda:
            "single_cell"
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/tables/keras-grid_search

            python bin/005a-scanpy_cluster_optimise_model-keras.py \
                --tissue {wildcards.tissue} \
                --sparse_matrix_path {input[1]} \
                --clusters_df {input[0]} \
                --genes_f {input[2]} \
                --cells_f {input[3]} \
                --number_epoch 25 \
                --batch_size 32 \
                --train_size_fraction 0.67 \
                --output_file results/{wildcards.tissue}/tables/keras-grid_search/keras-
            """
else:
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


# Define clustering resolutions
clustering_resolutions = list(config["clustering_resolutions"])

rule test_clusters_keras:
    input:
        "results/{tissue}/tables/clustering_array/leiden_{clustering_resolution}/adata_PCAd_batched_umap_{clustering_resolution}.h5ad",
        "results/{tissue}/tables/batch_correction/best_batch_method.txt",
        "results/{tissue}/tables/keras-grid_search/keras-use_params.txt",
        "results/{tissue}/objects/adata_unfilt_log1p_cp10k.h5ad"
    output:
        "results/{tissue}/tables/clustering_array/leiden_{clustering_resolution}/base-model_report.tsv.gz"
    resources:
        mem= lambda wildcards: 1000000 if wildcards.tissue == 'all_Epithelial' else 400000,
        queue= lambda wildcards: 'teramem' if wildcards.tissue == 'all_Epithelial' else 'normal',
        mem_mb=lambda wildcards: 1000000 if wildcards.tissue == 'all_Epithelial' else 400000,
        mem_mib=lambda wildcards: 1000000 if wildcards.tissue == 'all_Epithelial' else 400000,
        disk_mb=lambda wildcards: 1000000 if wildcards.tissue == 'all_Epithelial' else 400000,
        tmpdir="tmp",
        threads=16 # All 16
    conda:
        "single_cell"
    shell:
        r"""
        mkdir -p results/{wildcards.tissue}/tables/clustering_array
        mkdir -p results/{wildcards.tissue}/tables/clustering_array/leiden_{wildcards.clustering_resolution}
        python bin/005a-scanpy_cluster_optimise_model-keras_adata.py \
            --h5_anndata {input[0]} \
            --leiden_res {wildcards.clustering_resolution} \
            --number_epoch 25 \
            --keras_param_file {input[2]} \
            --batch_size 32 \
            --cluster_col "leiden" \
            --train_size_fraction 0.67 \
            --output_file results/{wildcards.tissue}/tables/clustering_array/leiden_{wildcards.clustering_resolution}/base \
            --full_h5 {input[3]}
        """


## Define function to aggregate the output of the clustering array results within tissue
def gather_within_tissue(wildcards):
    return expand("results/{tissue}/tables/clustering_array/leiden_{clustering_resolution}/base-model_report.tsv.gz",
                  tissue=wildcards.tissue,
                  clustering_resolution=config["clustering_resolutions"])

# Run the aggregation of the keras tests
rule cluster_qc_summarise_keras:
    input:
        gather_within_tissue
    output:
        "results/{tissue}/objects/adata_clusters_post_clusterQC.h5ad",
    params:
        max_proportion=config["max_proportion"],
        min_ncells=config["min_ncells"],
        MCC_thresh=config["MCC_thresh"],
        filter_based_on_prev_fails=config["filter_based_on_prev_fails"],
        consistent_failing_cluster_proportion=config["consistent_failing_cluster_proportion"],
        num_consistent_failing_clusters=config["num_consistent_failing_clusters"],
        use_cluster_QC=config["use_cluster_QC"]
    conda:
        "scvi-env"
    resources:
        mem=200000, # All - 350000
        queue='normal', # All - normal
        mem_mb=200000,
        mem_mib=200000,
        disk_mb=200000,
        tmpdir="tmp",
        threads=4
    shell:
        r"""
        mkdir -p results/{wildcards.tissue}/figures/clustering_array_summary
        mkdir -p results/{wildcards.tissue}/figures/UMAP/annotation
        python bin/006a-qc_clusters_keras.py \
            --tissue {wildcards.tissue} \
            --max_proportion {params.max_proportion} \
            --min_ncells {params.min_ncells} \
            --MCC_thresh {params.MCC_thresh} \
            --filter_based_on_prev_fails {params.filter_based_on_prev_fails} \
            --consistent_failing_cluster_proportion {params.consistent_failing_cluster_proportion} \
            --num_consistent_failing_clusters {params.num_consistent_failing_clusters} \
            --use_cluster_QC {params.use_cluster_QC}

        """

'''
# Define function to aggregate these across lineages and output to a single file
def gather_accross_tissues(wildcards):
    return expand("results/{tissue}/objects/adata_clusters_post_clusterQC.h5ad",
                  tissue=config["tissue"])

# Concatonate tissues
import yaml
configfile_yaml = open("config_cluster_within_lineage.yaml").read()
config_load = yaml.load(configfile_yaml, Loader=yaml.FullLoader)
tissues = config_load['tissue']
concatenated_values = ",".join(tissues)

rule combine_adatas_original:
    input:
        gather_accross_tissues
    output:
        "results/combined/objects/adata_grouped_post_cluster_QC.h5ad"
    conda:
        "scvi-env"
    params:
        base_tissue_original_h5=config["base_tissue_original_h5"],
        concatenated_values=concatenated_values,
        remove_blacklist=config["remove_blacklist"]
    resources:
        mem=650000, # All - 350000
        queue='normal', # All - normal
        mem_mb=650000,
        mem_mib=650000,
        disk_mb=650000,
        tmpdir="tmp",
        threads=4
    shell:
        r"""
        mkdir -p results/combined/{{figures,objects,tables}}
        mkdir -p results/combined/figures/UMAP

        python bin/006b-combine_across_lineages.py \
            --baseout "combined" \
            --base_tissue_original_h5 {params.base_tissue_original_h5} \
            --tissues "{params.concatenated_values}" \
            --remove_blacklist {params.remove_blacklist}
        """
            
rule make_lineage_prediction_model:
    input:
        "results/combined/objects/adata_grouped_post_cluster_QC.h5ad"
    output:
        "results/combined/tables/lineage_model/base-model_report.tsv.gz"
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
        mem=1000000,
        queue='teramem',
        mem_mb=1000000,
        mem_mib=1000000,
        disk_mb=1000000,
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


rule add_celltypist:
    input:
        "results/combined/objects/adata_grouped_post_cluster_QC.h5ad",
        "results/combined/tables/batch_correction/best_batch_method.txt"
    output:
        "results/combined/tables/annotation/CellTypist/CellTypist_prob_matrix.csv",
        "results/combined/tables/annotation/CellTypist/CellTypist_decision_matrix.csv",
        "results/combined/tables/annotation/CellTypist/CellTypist_anno_conf.csv"
    params:
        model=config["celltypist_model"],
        model_name=config["celltypist_model_name"]
    resources:
        mem=increment_memory(300000), #All - 850000
        queue='normal',
        mem_mb=increment_memory(300000),
        mem_mib=increment_memory(300000),
        disk_mb=increment_memory(300000),
        tmpdir="tmp",
        threads=16 
    conda:
        "scvi-env"
    shell:
        r"""
        mkdir -p results/combined/figures/annotation
        mkdir -p results/combined/tables/annotation
        mkdir -p results/combined/tables/annotation/CellTypist
        
        python bin/007-CellTypist.py \
            --tissue "combined" \
            --h5_path {input[0]} \
            --pref_matrix {input[1]} \
            --model {params.model} \
            --model_name {params.model_name}
        """


rule predict_all_cells:
    input:
        "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined/input/adata_raw_input_{tissue}.h5ad",
        "results/{tissue}/tables/optim_resolution.txt"
    output:
        "results/{tissue}/tables/annotation/keras/optimum_res_all_cells_all_probabilities.txt"
    singularity:
        "/software/hgi/containers/yascp/yascp.cog.sanger.ac.uk-public-singularity_images-wtsihgi_nf_scrna_qc_6bb6af5-2021-12-23-3270149cf265.sif"
    resources:
        mem=300000,
        queue='normal',
        mem_mb=300000,
        mem_mib=300000,
        disk_mb=300000,
        tmpdir="tmp",
        threads=2
    shell:
        r"""
        python bin/008-predict_all_probabilities_keras.py \
            --tissue "combined" \
            --sparse_matrix_path {input[0]} \
            --genes_f {input[1]} \
            --cells_f {input[2]} \
            --genes_var_f {input[3]} \
            --optim_resolution_f {input[4]} \
            --output_file "results/combined/tables/annotation/keras/optimum_res_all_cells_"
        """

rule add_predictions_filter:
    input:
        "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/input_test/adata_raw_input_{tissue}.h5ad",
        "results/{tissue}/tables/annotation/keras/optimum_res_all_cells_all_probabilities.txt"
    output:
        "results/{tissue}/objects/adata_raw_predicted_celltypes_filtered.h5ad"
    conda:
        "scvi-env_reserve"
    params:
        probability_threshold=config["probability_threshold"]
    resources:
        mem=300000,
        queue='normal',
        mem_mb=300000,
        mem_mib=300000,
        disk_mb=300000,
        tmpdir="tmp",
        threads=2
    shell:
        r"""
        python bin/009-add_all_predictions_adata_filter.py \
            --tissue {wildcards.tissue} \
            --fpath {input[0]} \
            --prediction_file {input[1]} \
            --probability_threshold {params.probability_threshold}
        """
'''
