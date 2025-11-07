"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



add symmetric

Appends symmetric solutions for the ground truth to the list of ground truth matrices 
in the .hdf5 files.

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - .hdf5 file with appended ground truth list
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import h5_files.h5_writer as w
import h5_files.file_reader as r
import numpy as np

"""
=============================================================================
------------------------------------MAIN-------------------------------------
=============================================================================
"""

def main(file_name, corr_angle, axis, corr_angle2=0, axis2=0):
    
    # Appending arrays to h5 file, source: https://stackoverflow.com/a/47074545
    
    " Read h5 file content "
    Told = r.h5reader(file_name,"transformation")
    Rold = Told[:,0:3,0:3]
    told = Told[:,0:3,3]
    
    " Create new matrix "
    Rcorr = w.rotation_matrix(corr_angle,axis)
    if(axis2 != 0):
        Rcorr2 = w.rotation_matrix(corr_angle2,axis2)
        Rcorr = np.matmul(Rcorr,Rcorr2)
    # print(Rcorr)
    Rnew = np.matmul(Rold,Rcorr)
    
    Tnew = w.homogenous_transfo(Rnew,told)
    
    w.append_h5(file_name,"transformation_all",Tnew)

if __name__ == '__main__':
    main()