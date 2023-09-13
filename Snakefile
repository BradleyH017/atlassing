rule preprocess_batch:
    input: 
        script = "atlassing_rectum_all_samples.py"
    output: 
        statpath = "rectum/healthy/All_not_gt_subset/objects/adata_PCAd_scvid.h5ad"
    params: 
        o = "logs/rectum_atlassing01-%J-output.log"
        e = "logs/rectum_atlassing01-%J-error.log"
        q = "long"
        J = "rectum_atlassing01"
    shell:
        """
        # Activate sc4 directory
        conda activate sc4
        # Submit job
        bsub -o {params.o} -e {params.e} -q {params.q} -J {params.J} < {input.script}
        """