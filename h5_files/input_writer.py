"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



Input Writer

Write correct files to evaluate PCR methods.

Inputs:
    - .stl (CAD) File   (Template)
    - .ply file         (Source)
    - Ground Thruth 

Output:
    - .hdf5 file containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Thruth

"""
"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import os
import math
import torch
import numpy as np
import open3d as o3d

#h5_files imports
import h5_files.h5_writer as w
import h5_files.file_reader as r

"""
=============================================================================
-----------------------VARIABLES (only thing to change)----------------------
=============================================================================
"""

# General
BASE_DIR = os.getcwd() #Parent folder -> Thesis
#print(BASE_DIR)

# CAD FILES NAMES
# CAD_name = "Base-Top_Plate"
# CAD_name = "Pendulum"
# CAD_name = "Round-Peg"
# CAD_name = "Separator"
# CAD_name = "Shaft_New"
# CAD_name = "Square-Peg"


"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""

def downsample_pcd(pcd_array,voxel_size):
    # :: Downsamples point cloud with given voxel size
    
    # Turn point cloud array into point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    
    # Downsample with given voxelsize
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down_array =  np.asarray(pcd_down.points)
    
    return pcd_down_array

def main(source_file, CAD_name, Y_angle, Z_angle, Plane_Normal_Vector, X_angle_correction = 0,
         translation_correction = [0,0,0], Normals_boolean = False, Folder_name = "Original",
         Normals_radius = 0.01, Normals_Neighbours = 30, scale = 100, multiple = 1, 
         result_file = "test", remove_mean = False, voxel_size = 0):

    """
    =============================================================================
    --------------------------------LOAD SOURCE----------------------------------
    =============================================================================
    """
    
    source_filename = BASE_DIR + "/h5_files/output/processing_results/" + source_file + ".txt"
    # Turn .txt file to array
    source_array = r.txt_to_array(source_filename)
    
    if(voxel_size != 0):
        source_array = downsample_pcd(source_array, voxel_size)
    
    # Dimension (3 if no normals, 6 with normals)
    DIM = source_array[1].size
    
    if(DIM == 6):
        Normals_boolean = True
    else:
        Normals_boolean = False
    
    # Array to tensor
    source_tensor = torch.from_numpy(source_array)
    source_tensor = source_tensor.expand(1,source_tensor.size(0),DIM)
    
    # Source mean
    source_mean = torch.mean(source_tensor,  1) 
    
    if(remove_mean):
        # Remove mean from PC to center on (0,0,0)
        source_tensor = source_tensor - source_mean                                                  
                                                                           
    source_nmb_points = source_tensor.shape[1]
    
    # print(source_tensor.size())

    """
    =============================================================================
    -------------------------------LOAD TEMPLATE---------------------------------
    =============================================================================
    """
    
    # Read Mesh file
    template_filename = BASE_DIR + "/datasets/CAD/"+ Folder_name + "/" + CAD_name + ".stl"
    template_mesh = o3d.io.read_triangle_mesh(template_filename)  
    
    # Sample Mesh uniformly
    templ_nmb_points = source_nmb_points*multiple
    templ_pcd = template_mesh.sample_points_uniformly(number_of_points=templ_nmb_points)  
    
    # Scale to get correct dimensions
    templ_array = np.asarray(templ_pcd.points)/scale
    
    # Estimate normals if normal information included
    if(Normals_boolean):
        templ_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=Normals_radius, max_nn=Normals_Neighbours))
        templ_normals = np.asarray(templ_pcd.normals)
        templ_array = np.concatenate((templ_array,templ_normals),1)
        
    # Visualize PC and Mesh
    o3d.visualization.draw_geometries([templ_pcd,template_mesh])                          
    
    # Turn PC to tensor of points
    templ_tensor = torch.from_numpy(templ_array)                                                            
                                                                
    templ_tensor = templ_tensor.expand(1,templ_nmb_points,DIM)
    
    templ_mean = torch.mean(templ_tensor, 1)                                           
    templ_tensor = templ_tensor - templ_mean   
    
    # print(templ_tensor.size())
    """
    =============================================================================
    ---------------------------DEFINE GROUND THRUTH------------------------------
    =============================================================================
    """
    # Translation vector, Ps = Pt + t
    translation_vector = torch.tensor(translation_correction)   
    if(not remove_mean):
        translation_vector = translation_vector + source_mean[:,0:3]                                                    
    
    # Angles
    Plane_Normal_Norm = np.linalg.norm(np.array(Plane_Normal_Vector))

    X_angle = math.acos(Plane_Normal_Vector[1]/Plane_Normal_Norm)*180/math.pi
    print("Computed angle over x-axis is %.3f.\n" % X_angle)
    X_angle = X_angle + X_angle_correction
    
    # Rotation matrix,    Ps = R*Pt
    Rz = w.rotation_matrix(Z_angle,"z")
    Rx = w.rotation_matrix(X_angle,"x")
    Ry = w.rotation_matrix(Y_angle,"y")
    
    # Multiply rotation matrices on rigth for rotation over new axis
    Rtemp = np.matmul(Rx,Rz)
    Rtot = np.matmul(Rtemp,Ry)
    
    # Construct total transformation matrix                                         
    Ground_truth = w.homogenous_transfo(Rtot,translation_vector)
    Ground_truth = np.expand_dims(Ground_truth,0)
    
    """
    =============================================================================
    ----------------------TRANSFORMED SOURCE (GROUND TRUTH)----------------------
    =============================================================================
    """
    
    transf_tmpl_tensor = w.apply_transfo(templ_tensor,Rtot,translation_vector,Normals_boolean)
    
    """
    =============================================================================
    ---------------------------------VISUALISE-----------------------------------
    =============================================================================
    """
    # Template (CAD) & Source (Scan)
    # r.show_open3d(templ_tensor[:,:,0:3], source_tensor[:,:,0:3], name = "Original Template (CAD) & Source (Scan)")
    
    # Transformed source (with GT) & Source (Scan)
    r.show_open3d(source_tensor[:,:,0:3], transf_tmpl_tensor[:,:,0:3], name="Ground Truth Transformed Template & Source (scan)")
    
    """
    =============================================================================
    --------------------------------WRITE FILE-----------------------------------
    =============================================================================
    """
    
    
    w.write_h5(result_file,templ_tensor,source_tensor,Ground_truth, FolderName = "experiments")

if __name__ == '__main__':
    main()
