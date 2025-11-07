"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



file reader

Read different types of files (.txt, .hdf5, .stl) and turn into array

Inputs:
    - .hdf5 files 
    - .txt files
    - .stl files

Output:
    - array with data
    
Credits:
    .hdf5 file reading
    LINK: https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
    
    .txt file reading
    LINK: https://stackoverflow.com/questions/14676265/how-to-read-a-text-file-into-a-list-or-an-array-with-python


"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import h5py
import open3d as o3d
import numpy as np
import torch

"""
=============================================================================
-----------------------------READING FUNCTIONS-------------------------------
=============================================================================
"""
def read_stl(file_loc, Nmb_points, Normals = False, Center = True, Scale = True):   
    # :: Read .stl file, scale and estimate normals
    
    # Create point cloud
    mesh = o3d.io.read_triangle_mesh(file_loc)                       # Read Mesh file
    pcd = mesh.sample_points_uniformly(number_of_points=Nmb_points)  # Sample Mesh uniformly
    
    # Create array + scale + center
    data_array = np.asarray(pcd.points)                              # Turn PC to array of points
    
    data_mean = np.mean(data_array,0)                                # Calculate PC mean
    
    if(Center):
        data_array = data_array - data_mean                          # Remove mean from PC to center
                                                                     # on (0,0,0)
    max_value = 1
    if(Scale):
        max_value = np.max(np.max(np.abs(data_array),0))
        #print(max)
        data_array = data_array/(max_value)
    
    # Add normals if required
    if(Normals):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        normals_array = np.asarray(pcd.normals)
        data_array = np.append(data_array,normals_array,1)
    
    return data_array, data_mean, max_value


def read_Test_txt(file_loc): 
    # -- unused --
    # :: Read estimated transformation from .txt file               
    # :: Test should be in the form of a 2D array
    
    file = open(file_loc,'r')
    T_list = []
    T_est = []
    lines = file.readlines()
    count = 0
    
    for line in lines:
                                            # Check for start of new matrix (indicated by '[[')
        if(line[0:1] == "["):
            L = line.split('[[')[-1]        # Split line at '[[' and keep only last element from list
            L = L.split(']')[0]             # Split line at ']' and keep only first element from list
            L = L.split(' ')                # Split at spaces 
            L = [x for x in L if x != '']   # Remove empty characters from list
            L = [float(x) for x in L]       # Transfrom strings to floats
            # print(line)
            # print(L)
            
            T_est.append(L)                 # Add line to new list 
            count = count + 1               # Increase count
            
        if(line[0] == " "):                 # Check for other lines of matrix (indicated by ' ')
            L = line.split('[')[-1]         # Split line at '[' and keep only last element from list
            L = L.split(']')[0]             # Split line at ']' and keep only first element from list
            L = L.split(' ')                # Split at spaces
            L = [x for x in L if x != '']   # Remove empty characters from list
            L = [float(x) for x in L]       # Transform strings to floats
            # print(line)
            # print(L)
            
            T_est.append(L)                 # Add line to new list
            count = count + 1               # Increase count
            
        if(count == 4):                     # When count is 4, matrix is finished (1x4x4)
            T_list.append(T_est)            # Add found matrix to list
            T_est = []                      # Reset list for new matrix
            count = 0                       # Reset count
    
    # Turn list to tensor (correct format)
    transfo_array = np.array(T_list)
    return transfo_array


def h5reader(file_dict,key):
    # Read h5 file from directory with given key
    # :: Link: https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
    
    with h5py.File(file_dict,"r") as f:
        if(key in f.keys()):
            dataset = f[key][()]
        else:
            raise Exception("Key not included in %s" % f.keys())
    return dataset

def txt_to_array(filename):
    # :: Read .txt file and turn into array
    # Link: https://stackoverflow.com/questions/14676265/how-to-read-a-text-file-into-a-list-or-an-array-with-python
    
    list_of_lists = []

    with open(filename) as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split(' ')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            list_of_lists.append(inner_list)

    array = np.array(list_of_lists)
    return array

"""
=============================================================================
-----------------------------DATALOADER IMPORT-------------------------------
=============================================================================
"""
# Import dataloader function
from misc import dataloader

def h5file_to_torch(file_loc, zero_mean, T_est = False):
    # :: Turn basic .hdf5 file, containing template, source, ground truth and estimated transformation
    # :: into tensor
    
    dataset = dataloader.dataset_loader(file_loc, zero_mean, Test = T_est)
    templ_array = dataset.template
    src_array = dataset.source
    gt_array = dataset.transformation
    gt_symm_array = dataset.transformation_all
    
    templ_tensor = torch.tensor(templ_array)
    src_tensor = torch.tensor(src_array)
    gt_tensor = torch.tensor(gt_array)
    gt_symm_tensor = torch.tensor(gt_symm_array)
    
    if(T_est):
        transfo_array = dataset.estimated_transform
        transfo_tensor = torch.tensor(transfo_array,dtype=torch.float64)
        return templ_tensor, src_tensor, gt_tensor, gt_symm_tensor, transfo_tensor
    
    # print(templ_tensor.shape)
    # print(src_tensor.shape)
    # print(gt_tensor.shape)
    
    return templ_tensor, src_tensor, gt_tensor, gt_symm_tensor

"""
=============================================================================
------------------------------DISPLAY & OTHER--------------------------------
=============================================================================
"""

def show_open3d(template,source, name = "Open3D", index = 0):
    # :: Visualise template and source
    
    # Create point clouds
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template[index][:][:])
    source_.points = o3d.utility.Vector3dVector(source[index][:][:])
    
    # Add color
    template_.paint_uniform_color([0,0,1])
    source_.paint_uniform_color([1,0,0])
    
    #bounding box template/source
    templ_bb = template_.get_oriented_bounding_box()
    templ_bb.color = (1, 0, 0)
    
    src_bb = source_.get_oriented_bounding_box()
    src_bb.color = (0,1,0)
    
    # Add axis system
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    # o3d.visualization.draw_geometries([template_,source_,templ_bb,src_bb])
    # o3d.visualization.draw_geometries([template_,source_,mesh_frame],window_name=name)
    o3d.visualization.draw_geometries([template_,source_],window_name=name)
    # volume_ratio(templ_bb, src_bb)

def volume_ratio(bb_1,bb_2):
    # :: Compute  ratio of volumes for rectangles
    corners_1 = np.array(bb_1.get_box_points())
    corners_2 = np.array(bb_2.get_box_points())
    
    V1 = rect_volume(corners_1)
    print(":: Template volume is: ",V1)
    V2 = rect_volume(corners_2)
    print(":: Source volume is: ",V2)
    
    ratio = V2/V1
    print(":: The volume ratio of source & template is: ",ratio)
    return

def rect_volume(corners):
    # :: Compute rectangle volume
    l1 = np.linalg.norm(corners[1]-corners[0])
    l2 = np.linalg.norm(corners[2]-corners[0])
    l3 = np.linalg.norm(corners[3]-corners[0])
    V = l1*l2*l3
    print(V)
    return V