#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
    
# Define functions:
########### Define custom functions used here ###############
def nmad_calc(data):
    import numpy as np
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    # Use direction aware filtering
    nmads = (data - np.median(data))/mad
    return(nmads)

def nmad_append(df, var, group=[]):
    import pandas as pd
    import numpy as np
    # Calculate
    if group:
        vals = df.groupby(group)[var].apply(nmad_calc)
        # Return to df
        temp = pd.DataFrame(vals)
        temp.reset_index(inplace=True)
        if "level_1" in temp.columns:
            temp.set_index("level_1", inplace=True)
        else: 
            index_name=df.reset_index().columns[0]
            temp.set_index(index_name, inplace=True)
        
        #temp = temp[var]
        temp = temp.reindex(df.index)
        return(temp[var])
    else:
        vals = nmad_calc(df[var])
        return(vals)

def dist_plot(adata, column, within=None, relative_threshold=None, absolute=None, out="./", thresholded = False, threshold_method = None, out_suffix = "", relative_directionality = None):
    """
    Plots the distribution and relative threshold cut offs for a value of relative_threshold
    
    Parameters:
    adata: 
        anndata object
    column: 
        column of anndata object to compute relative scores and filtration of. Must be numeric
    within: 
        Grouping variable of anndata.obs. For example, would be "tissue" if scores are to be calculated within groups of adata.obs['tissue']
        If none are desired, use a dummy column that is consistent for all rows of adata.obs
    relative_threshold: 
        Numeric value indicating the number of MAD from the median that threshold should be applied from.
    absolute: 
        Optional. Is there also an absolute threshold to be used?
        This can be a dictionary if different absolute thresholds are to be applied to different values of 'within', each element's name must perfectly match the levels of groups.
    out: 
        Directory for plots to be saved in.
    out_suffix:
        Suffix to add to output file name. Name structure will be f"{out}/{column}_per_{within}{out_suffix}.png"
    thresholded: 
        Has the data already been thresholded? If so, there must be a boolean column called adata.obs[f"{column}_keep"] which indicates whether a cell is kept on the basis of this column
    threshold_method:
        Which threshold method to use. Can be one of "specific" or "outer".
        "specific" will apply thresholds that are specific to the grouping of "within"
        "outer" will apply the outermost thresholds from each of the groupings of "within"
    relative_directionality:
        Specifies directionality of relative thresholds. Only used when actually thresholding
    
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    # Function to plot and calculate the distribution
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    if within is not None:
        groups = np.unique(adata.obs[within])
        for g in groups:
            data = adata.obs.loc[adata.obs[within] == g, column].values
            absolute_diff = np.abs(data - np.median(data))
            mad = np.median(absolute_diff)
            line_color = sns.color_palette("tab10")[int(np.where(groups == g)[0])]
            if thresholded:
                if threshold_method == "specific":
                    # Calculate the threshold actually being used for this group
                    cutoff_low = min(adata.obs[column][adata.obs[f"{column}_keep"] == True])
                    cutoff_high = max(adata.obs[column][adata.obs[f"{column}_keep"] == True])
                else:
                    # Calculate the threshold actually being used across all groups
                    cutoff_low = min(adata.obs[column][adata.obs[f"{column}_keep"] == True])
                    cutoff_high = max(adata.obs[column][adata.obs[f"{column}_keep"] == True])
            else:
                # Calculate the relative threshold being used for this group of data
                cutoff_low = np.median(data) - (float(relative_threshold) * mad)
                cutoff_high = np.median(data) + (float(relative_threshold) * mad)
            if relative_threshold is not None:
                sns.distplot(data, hist=False, color = line_color, rug=True, label=f'{g} (relative): {cutoff_low:.2f}-{cutoff_high:.2f}')
                plt.axvline(x = cutoff_low, linestyle = '--', color = line_color, alpha = 0.5)
                plt.axvline(x = cutoff_high, linestyle = '--', color = line_color, alpha = 0.5)
            else:
                sns.distplot(data, hist=False, color = line_color, rug=True, label=f'{g}')
    #
    else:
        data = adata.obs[column]
        absolute_diff = np.abs(data - np.median(data))
        mad = np.median(absolute_diff)
        line_color = sns.color_palette("tab10")[int(np.where(groups == g)[0])]
        if thresholded:
            # Calculate the threshold actually being used
            cutoff_low = min(adata.obs[column][adata.obs[f"{column}_keep"] == True])
            cutoff_high = max(adata.obs[column][adata.obs[f"{column}_keep"] == True])
        else:
            # Calculate the relative threshold being used for this group of data
            cutoff_low = np.median(data) - (float(relative_threshold) * mad)
            cutoff_high = np.median(data) + (float(relative_threshold) * mad)
        if relative_threshold is not None:
            sns.distplot(data, hist=False, rug=True, label=f'relative: {cutoff_low:.2f}-{cutoff_high:.2f}')
        else:
            sns.distplot(data, hist=False, rug=True)
    plt.legend()
    plt.xlabel(column)
    if absolute is not None:
        if isinstance(absolute, dict):
            temp = [value for value in absolute.values() if isinstance(value, (int, float))]
            temp = np.unique(temp)
            plt.title(f"Absolute (black): {absolute}. Threshold_method = {threshold_method}, directionality = {relative_directionality}")
        else:
            temp = [absolute]
            plt.title(f"Absolute (black): {absolute:.2f}, Threshold_method = {threshold_method}, directionality = {relative_directionality}")
        for a in temp:
            plt.axvline(x = a, color = 'black', linestyle = '--', alpha = 0.5)
    #
    else:
        plt.title(f"No absolute cut off")
    plt.savefig(f"{out}/{column}_per_{within}{out_suffix}.png", bbox_inches='tight')
    plt.clf()


def update_obs_qc_plot_thresh(adata, column = None, within = None, relative_threshold = None, threshold_method = "specific", relative_directionality = "bi", absolute = None, absolute_directionality = "over", plot = True, out = "./", out_suffix = ""):
    """
    Updates the .obs of an anndata object with median absolute deviation scores and plots proposed filters based on these
    
    Parameters:
    adata: 
        anndata object
    column: 
        column of anndata object to compute relative scores and filtration of. Must be numeric
    within: 
        Grouping variable of anndata.obs. For example, would be "tissue" if scores are to be calculated within groups of adata.obs['tissue']
        If none are desired, use a dummy column that is consistent for all rows of adata.obs
    relative_threshold: 
        Numeric value indicating the number of MAD from the median that threshold should be applied from.
    threshold_method: 
        Which threshold method to use. Can be one of "specific", "outer" or None
        "specific" will apply thresholds that are specific to the grouping of "within"
        "outer" will apply the outermost thresholds from each of the groupings of "within".
        None will not apply the relative threshold after calculation
        If "outer" is used in conjunction with unidirectionality (over/under), the max/min threshold will be applied in that direction only
    relative_directionality: 
        What is the directionality of the relative threshold to be used? Can be one of "bi", "over" or "under" for bidirectional, unidirectional keep cells over or unidirectional keep cells under respectively
    absolute: 
        Optional. Is there also an absolute threshold to be used? 
        This can be a dictionary if different absolute thresholds are to be applied to different values of 'within', each element's name must perfectly match the levels of groups.
    absolute_directionality: 
        What is the directionality of cells to keep relative to the absolute threshold? Can be "over" or "under".
    plot: 
        Plot the results?
    out: 
        Directory for plots to be saved in.
    out_suffix:
        Suffix to add to output file name. Name structure will be f"{out}/{column}_per_{within}{out_suffix}.png"
    """
    print(f"----- Calculating thresholds for: {column} ------")
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Calculate the relative threshold per value of within
    adata.obs[f"{column}_nMAD"] = nmad_append(adata.obs, column, within)
    if threshold_method == "specific":
        print(f"Applying {within}-specific nMAD thresholds")
        # Add boolean vector specific to each group
        if relative_directionality == "bi":
            adata.obs[f"{column}_keep"] = abs(adata.obs[f"{column}_nMAD"]) < relative_threshold # Bidirectional
        if relative_directionality == "over":
            adata.obs[f"{column}_keep"] = adata.obs[f"{column}_nMAD"] > relative_threshold
        if relative_directionality == "under":
            adata.obs[f"{column}_keep"] = adata.obs[f"{column}_nMAD"] < relative_threshold
    # determine how thresholds should be applied
    if threshold_method == "outer":
        print(f"Applying outer window of {within}-specific nMAD thresholds")
        # Generate a vector of thresholds
        thresh = []
        groups = np.unique(adata.obs[within])
        for g in groups:
            data = adata.obs.loc[adata.obs[within] == g,column].values
            absolute_diff = np.abs(data - np.median(data))
            mad = np.median(absolute_diff)
            cutoff_low = np.median(data) - (float(relative_threshold) * mad)
            cutoff_high = np.median(data) + (float(relative_threshold) * mad)
            thresh.append([cutoff_low, cutoff_high])
        # Flatten this list to find max / min
        flattened_list = [item for sublist in thresh for item in sublist]
        max_value = max(flattened_list)
        min_value = min(flattened_list)
        if relative_directionality == "bi":
            adata.obs[f"{column}_keep"] = (adata.obs[column] > min_value) & (adata.obs[column] < max_value) # Bidirectional
        if relative_directionality == "over":
            adata.obs[f"{column}_keep"] = adata.obs[column] > min_value
        if relative_directionality == "under":
            adata.obs[f"{column}_keep"] = adata.obs[column] < max_value
    if threshold_method == "None":
        # Don't apply any relative threshold after calculating. Make dummy for all TRUE
        adata.obs[f"{column}_keep"] = True
    # Add absolute threshold if using
    if absolute is not None:
        # If applying multiple absolute thresholds to each level of within separately
        if isinstance(absolute, dict):
            print(f"Applying {within}-specific absolute thresholds")
            if absolute_directionality == "over":
                for g in absolute.keys():
                    adata.obs.loc[(adata.obs[within] == g) & (adata.obs[column] < absolute[g]), f"{column}_keep"] = False
            if absolute_directionality == "under":
                for g in absolute.keys():
                    adata.obs.loc[(adata.obs[within] == g) & (adata.obs[column] > absolute[g]), f"{column}_keep"] = False
        else:
            print(f"Not applying {within}-specific absolute thresholds")
            if absolute_directionality == "over":
                adata.obs.loc[adata.obs[column] < absolute, f"{column}_keep"] = False
            if absolute_directionality == "under":
                adata.obs.loc[adata.obs[column] > absolute, f"{column}_keep"] = False
    if plot:
        print(f"Plotting to: {out}/{column}_per_{within}{out_suffix}.png")
        # Plot the distribution with the thresholds actually being used (NOTE: This will be dependent on relative grouping)
        dist_plot(adata, column, within=within, relative_threshold=relative_threshold, absolute=absolute, out=out, thresholded = True, threshold_method = threshold_method, out_suffix = out_suffix, relative_directionality = relative_directionality)
