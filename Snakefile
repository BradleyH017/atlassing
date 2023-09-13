rule preprocess_batch:
    input: 
        script = "atlassing_rectum_all_samples.py"
    output: 

    run: 
    params: 
        category = "All_not_gt_subset"
    shell:
        """
        {input.script} {params.category} 
        """