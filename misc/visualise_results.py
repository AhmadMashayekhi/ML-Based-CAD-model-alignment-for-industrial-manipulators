"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



visualise results

Visualises the point clouds before and after the registration process.

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth
    - .txt file containing estimated transformation

Output:
    - Open3D visualisation
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import torch
from h5_files.file_reader import h5file_to_torch, h5reader, show_open3d

"""
=============================================================================
------------------------------------CODE-------------------------------------
=============================================================================
"""

def transform_source_tensor(src_tensor,transfo_tensor):
    
    # Extract rotation matrix and translation vector
    R_est = transfo_tensor[:, 0:3, 0:3]                          
    t_est = transfo_tensor[:, 0:3, 3].unsqueeze(2)     
    R_est_inv = R_est.permute(0, 2, 1)                        
    
    # Compute transformed source
    transf_src_tensor = torch.add(torch.bmm(src_tensor,R_est_inv),t_est.transpose(2,1))
    
    return transf_src_tensor
    
    
def main(h5_file_loc,zero_mean = False):
    
    # Load data
    templ_tensor, src_tensor, gt_tensor, gt_symm_tensor, transfo_tensor = h5file_to_torch(h5_file_loc, 
                                                                                          zero_mean,
                                                                                          T_est = True)
    
    # Compute transformed source
    transf_src_tensor = transform_source_tensor(src_tensor[:,:,0:3], transfo_tensor)
    trans_templ_tensor = transform_source_tensor(templ_tensor[:,:,0:3], gt_tensor)
    
    nmb_obj = templ_tensor.size(0)
    for i in range(nmb_obj):
        show_open3d(templ_tensor[:,:,0:3], src_tensor[:,:,0:3],
                      name="Original Template & Source",index=i)
        show_open3d(templ_tensor[:,:,0:3],transf_src_tensor[:,:,0:3], 
                      name="Registration Transformed Source & Template",index=i)
        show_open3d(trans_templ_tensor[:,:,0:3],src_tensor[:,:,0:3], 
                      name=" Ground Truth Transformed Template & Source",index=i)
    
if __name__ == '__main__':
    main()

