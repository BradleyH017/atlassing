configfile: "config_predict_all_cells.yaml"

rule all:
    input:
        expand("results/{tissue}/objects/keras_full_annot_not_subset.h5ad", tissue=config["tissue"])

if config["two_step_prediction"]:
    rule divide_per_sample:
        output:
            "results/{tissue}/objects/per_samp_adata/{sample}.h5ad"
        conda:
            "scvi-env"
        resources:
            mem=120000,
            queue='normal',
            mem_mb=120000,
            mem_mib=120000,
            disk_mb=120000,
            tmpdir="tmp",
            threads=2
        params:
            orig_h5ad=config["original_input_round1"],
            samp_col=config["batch_column"]
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/objects/per_samp_adata
            python bin/008b-divide_adata_per_samp.py \
                --h5ad {params.orig_h5ad} \
                --samp_col "convoluted_samplename" \
                --sample {wildcards.sample} \
                --output_file "results/{wildcards.tissue}/objects/per_samp_adata/"
            """


    rule predict_cells_1:
        input:
            "results/{tissue}/objects/per_samp_adata/{sample}.h5ad"
        output:
            "results/{tissue}/objects/keras/keras_annot_round1_{sample}.h5ad"
        conda:
            "single_cell"
        params:
            model=config["round1_model"],
            weights=config["round1_model_weights"]
        resources:
            mem=10000,
            queue='normal',
            mem_mb=10000,
            mem_mib=10000,
            disk_mb=10000,
            tmpdir="tmp",
            threads=2
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/objects/keras
            python bin/008-predict_all_probabilities_keras_adata.py \
                --h5ad {input} \
                --model {params.model} \
                --weights {params.weights} \
                --output_file "results/combined/objects/keras/keras_annot_round1_{wildcards.sample}"
            """

    rule stream_1:
        input:
            "results/{tissue}/objects/keras/keras_annot_round1_{sample}.h5ad"
        output:
            "results/{tissue}/objects/keras2/keras_annot_round1_{sample}_epithelial.h5ad",
            "results/{tissue}/objects/keras2/keras_annot_round1_{sample}_immune.h5ad",
            "results/{tissue}/objects/keras2/keras_annot_round1_{sample}_mesenchymal.h5ad"
        conda:
            "single_cell"
        resources:
            mem=15000,
            queue='normal',
            mem_mb=15000,
            mem_mib=15000,
            disk_mb=15000,
            tmpdir="tmp",
            threads=2
        shell:
            r"""
            mkdir -p results/{wildcards.tissue}/objects/keras2
            python bin/009-adjust_adata_filter_round1_prediction.py \
                --h5ad {input} \
                --col_to_sort "predicted_celltype" \
                --output_file "results/{wildcards.tissue}/objects/keras2/keras_annot_round1_{wildcards.sample}"
            """
        
    rule predict_cells_2:
        input:
            "results/{tissue}/objects/keras2/keras_annot_round1_{sample}_{model_subset}.h5ad",
            "results_round2/" + config["round2_prefix"] + "{model_subset}/tables/optim_resolution.txt"
        output:
            "results/{tissue}/objects/keras2/keras_annot_round2_{sample}_{model_subset}.h5ad"
        conda:
            "single_cell"
        params:
            round2_prefix=config["round2_prefix"]
        resources:
            mem=7000,
            queue='normal',
            mem_mb=7000,
            mem_mib=7000,
            disk_mb=7000,
            tmpdir="tmp",
            threads=1
        shell:
            r"""
            # Get the numeric value of resolution to use
            path="results_round2/{params.round2_prefix}{wildcards.model_subset}/tables"
            numeric_value=$(awk '{{printf "%.2f", $1}}' ${{path}}/optim_resolution.txt)
            # If this is "00" it won't match the file paths, replace
            if [[ $numeric_value == *".00" ]]; then
                numeric_value="${{numeric_value%.00}}.0"
            fi

            if [[ $numeric_value == *".50" ]]; then
                numeric_value="${{numeric_value%.50}}.5"
            fi


            # Get the model and weights path
            model_f="${{path}}/clustering_array/leiden_${{numeric_value}}/base.h5"
            weights_f="${{path}}/clustering_array/leiden_${{numeric_value}}/base-weights.tsv.gz"

            echo "leiden:" ${{numeric_value}}
            echo "model_f:" ${{model_f}}
            echo "weights_f:" ${{weights_f}}
            # Pass these to the script
            python bin/008-predict_all_probabilities_keras_adata.py \
                --h5ad {input[0]} \
                --model $model_f \
                --weights $weights_f \
                --output_file "results/{wildcards.tissue}/objects/keras2/keras_annot_round2_{wildcards.sample}_{wildcards.model_subset}"
            """
    
    # Read in samples and define function to gather
    import pandas as pd
    import yaml
    configfile_yaml = open("config_predict_all_cells.yaml").read()
    config_load = yaml.load(configfile_yaml, Loader=yaml.FullLoader)
    samp_file= config["samp_list"]
    sample_table = pd.read_table(samp_file, header=None)
    sample_table.columns = ["samp"]
    samples=list(sample_table.samp.unique())

    def gather_subsets(wildcards):
        return expand("results/{tissue}/objects/keras2/keras_annot_round2_{sample}_{model_subset}.h5ad",
                    tissue=config["tissue"],
                    sample=samples,
                    model_subset=config["round1_groups"])

    rule aggregate_after_rule_2:
        input:
            "results_round3/{tissue}/objects/adata_PCAd_batched_umap.h5ad", # Object with high confidence clusters (for intersection), from the previous round
            gather_subsets # output from the above filter
        params:
            orig_h5ad=config["original_input_round1"],
            samp_list=config["samp_list"]
        conda:
            "scvi-env"
        output:
            "results/{tissue}/objects/keras_full_annot_not_subset.h5ad"
        resources:
            mem=750000,
            queue='teramem',
            mem_mb=750000,
            mem_mib=750000,
            disk_mb=750000,
            tmpdir="tmp",
            threads=2
        shell:
            r"""
            python bin/010_combine_after_prediction.py \
                --orig_h5ad {params.orig_h5ad} \
                --highQC_h5ad {input[0]} \
                --round2_output "results/{wildcards.tissue}/objects/keras2/keras_annot_round2_" \
                --output_file "results/{wildcards.tissue}/objects/keras_full_annot_not_subset"
            """
#else:
    # Write the single-step prediction here
