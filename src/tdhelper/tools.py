# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def toFrequency(value_list, bins, limits):
    '''
    Calculate the frequency histogram for a list of values.
    
    Parameters
    ----------
    value_list: list
        List of values.
    bins: int
        Number of bins to create to generate a histogram.
    limits: tuple
        Tuple of (min, max) values to generate a histogram.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the bin centers and the frequency of values in each bin.

    '''

    bin_vals, bin_edges = np.histogram(value_list, bins=bins, range=limits)
    bin_vals = bin_vals / len(value_list)  # normalize to get relative frequency
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    
    return pd.DataFrame({'bin_centers': bin_centers, 'bin_vals': bin_vals})