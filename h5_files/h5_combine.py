"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



h5 combine

Combine different .h5 files into one file 

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - .hdf5 file containing all input .hdf5 files

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import os
import numpy as np
from tqdm import tqdm

#h5_files imports
import h5_files.h5_writer as w
import h5_files.file_reader as r

"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""

def combine_data(DIR):
    templ_list = []
    src_list = []
    gt_list = []
    
    # Loop over all files in directory
    for i,path in enumerate(tqdm(os.listdir(DIR))):
        
        # Check if current path is a file
        if os.path.isfile(os.path.join(DIR, path)):
            
            # Current file path
            file_path = os.path.join(DIR, path)
            
            # Extract information
            templ_array = r.h5reader(file_path,'template')
            src_array = r.h5reader(file_path,'source')
            gt_array = r.h5reader(file_path,'transformation')
            gt_array = np.expand_dims(gt_array,0) # Correct dimension: 1x4x4
            
            templ_list.append(templ_array.tolist())
            src_list.append(src_array.tolist())
            gt_list.append(gt_array.tolist())
        
    # print(templ_list.shape)
    # print(src_list.shape)
    # print(gt_list.shape)
    
    return templ_list, src_list, gt_list

def main(DIR, File_Name):
    
    templ_list, src_list, gt_list = combine_data(DIR)
    
    templ_list_arr = np.asarray(templ_list)
    src_list_arr = np.asarray(src_list)
    gt_list_arr = np.asarray(gt_list)
    w.write_h5(File_Name, templ_list_arr, src_list_arr, gt_list_arr, FolderName = "experiments/")
    
    
if __name__ == '__main__':
    main()