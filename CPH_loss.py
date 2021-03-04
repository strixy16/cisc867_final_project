# Name: CPH_loss.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: March 4, 2021

import numpy as np

def _make_riskset(time):
    """
    Compute mask that represents each sample's risk set
    
    Args:
        time - numpy array of observed event times, shape = (n_samples,) 
              
    
    Returns:
        risk_set - numpy array of boolean values with risk sets in rows, shape = (n_samples, n_samples)            
    """
    
    assert time.ndim == 1
    
    # Sort in descending order
    o = np.argsort(-time, kind="mergesort")
    
    # Initialize risk set
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    
    return risk_set
    
class InputFunction: 
    """
    Callable input function that computes the risk set for each batch.
    
    Args:
        images - numpy array of images, shape = (n_samples, height, width)
        time - numpy array of observed time labels, shape = (n_samples,)
        event - numpy array of event indicator, shape = (n_samples,)
        batch_size - int, number of samples per batch
        drop_last - int, whether to drop the last incomplete batch
        seed
    """