#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

import pandas as pd
import seaborn as sns
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
import numpy as np

def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Summarise NN results
                """
        )
    
    parser.add_argument(
            '-sf', '--summary_file',
            action='store',
            dest='summary_file',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-tissue', '--tissue',
            action='store',
            dest='tissue',
            required=True,
            help=''
        )
    
    return parser.parse_args()


# Define main script
def main():
    # Parse options
    inherited_options = parse_options()
    summary_file = inherited_options.summary_file
    tissue = inherited_options.tissue
    
    # Load the collated results
    res = pd.read_csv(summary_file, header=None)
    res = res.iloc[:,1:]
    res.columns = ["parameter", "value", "kNN"]

    # Plot each parameter as a line graph, kNN on x axis
    params = np.unique(res['parameter'])
    outdir=f"results/{tissue}/tables/nn_array"
    res['kNN'] = res['kNN'].astype(int)
    res = res.sort_values(by='kNN')

    for p in params:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        data = res[res['parameter'] == p]
        ax.plot(data['kNN'], data['value'],marker='o')
        plt.xlabel("kNN", fontsize=14)
        plt.ylabel(p, fontsize=14)
        ax.set_title(f"{p} across kNN values", fontsize=16)  # Adjust fontsize as needed
        plt.savefig(f"{outdir}/{p}_by_kNN.png", bbox_inches='tight')
        plt.clf()
    
    # Pick NN with max iLISI, or the smallest with median iLISI > 0.02
    
    

    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()