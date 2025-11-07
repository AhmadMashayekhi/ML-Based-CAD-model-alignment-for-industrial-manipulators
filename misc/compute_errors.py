"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



compute errors

Compute the different metrics based on the template, source, transformed source,
ground truth & estimated transformation.

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth
        o Estimated Transformation

Output:
    - Relative Errors
    - Root Mean Square Errors
    - Mean Absolute Errors
    - Recall
    - R2 
    - Chamfer Distance
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import time
import numpy as np
from misc.errors import Errors
from h5_files.file_reader import h5file_to_torch

"""
=============================================================================
------------------------------------CODE-------------------------------------
=============================================================================
"""

def compute_mean_variance(error_list):
    mean = np.mean(error_list,1)
    variance = np.var(error_list,1)
    return mean, variance

def symmetric_errors(templ_tensor, src_tensor, gt_symm_tensor, transfo_tensor, recall_lim):
    # Go over all possible ground truth solutions and find best one
    # Initialize error with max possible value (180°)
    max_recall = 0
    nmb_sol = gt_symm_tensor.shape[0]
    error_time = 0
    errors_min = np.ones((9,1))*-1
    # print(nmb_sol)
    
    for i in range(nmb_sol):
        # Compute errors (+ time)
        # Create error module
        
        gt_sol = gt_symm_tensor[i,:,:].expand(1,4,4)
        error_class = Errors(templ_tensor[:,:,0:3],src_tensor[:,:,0:3],gt_sol,transfo_tensor,recall_lim)
        
        # Compute errors
        start = time.time()
        errors = error_class()
        error_time = time.time()-start + error_time
        
        # Display errors
        # error_class.display(errors)
        
        # Save errors if smallest one
        if(errors[6,0] > max_recall):
            errors_min = errors
            max_recall = errors[6,0]
            
    mean_error_time = error_time/nmb_sol
    # Display errors
    # error_class.display(errors_min)
    return errors_min, mean_error_time

def non_symmetric_errors(templ_tensor, src_tensor, gt_tensor, transfo_tensor, recall_lim):
    error_class = Errors(templ_tensor[:,:,0:3],src_tensor[:,:,0:3],gt_tensor,transfo_tensor,recall_lim)
    
    # Compute errors
    start = time.time()
    errors = error_class()
    error_time = time.time()-start
    
    # Display errors
    # error_class.display(errors)

    return errors, error_time   

def main(h5_file_loc,recall_lim,symm_sol = False, zero_mean = False):
    
    # Load data
    templ_tensor, src_tensor, gt_tensor, gt_symm_tensor, transfo_tensor = h5file_to_torch(h5_file_loc, 
                                                                                          zero_mean,
                                                                                          T_est = True)
    
    # In case multiple solutions exist
    if(symm_sol):
        errors_min, mean_error_time = symmetric_errors(templ_tensor, src_tensor, gt_symm_tensor, 
                                                       transfo_tensor, recall_lim)
    else:
        errors_min, mean_error_time = non_symmetric_errors(templ_tensor, src_tensor, gt_tensor, 
                                                       transfo_tensor, recall_lim)
    
    # print(":: Mean error time is: ", mean_error_time)
    
    return errors_min
    
if __name__ == '__main__':
    main()
