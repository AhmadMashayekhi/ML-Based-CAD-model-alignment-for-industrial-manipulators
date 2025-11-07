"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



dataloader

Loads in the .hdf5 file and returns a dataset loader object with the desired
parameters to loop over all objects inside.

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth
        o Estimated Transformation

Output:
    - dataloader object
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
import torch
import numpy as np
import open3d as o3d
from h5_files.file_reader import h5reader
from torch.utils.data import DataLoader

"""
=============================================================================
-----------------------------------CLASS-------------------------------------
=============================================================================
"""

class dataset_loader():
    # :: Create class to load .hdf5 files with desired parameters
    def __init__(self,file_loc,zero_mean = False, normals = False, voxel_size = 0, Test = False):
        self.file_loc = file_loc
        self.template = h5reader(file_loc,'template')
        self.source = h5reader(file_loc,'source')
        self.transformation = h5reader(file_loc,'transformation')
        self.transformation_all = h5reader(file_loc,'transformation_all')
        self.normals = normals
        self.voxel_size = voxel_size
        self.Test = Test
        
        if(Test):
            self.estimated_transform = h5reader(file_loc,'Test')
        
        if(zero_mean):
            self.template,self.template_mean = remove_mean(self.template)
            self.source,self.source_mean = remove_mean(self.source)
            self.transformation = remove_mean_transformation(self.transformation, self.source_mean)
            self.transformation_all = remove_mean_transformation(self.transformation_all, self.source_mean)

        if(voxel_size != 0):
            self.template = self.downsample_pcd(self.template)
            self.source = self.downsample_pcd(self.source)
        
        if(normals):
            self.DIM = 6
        else:
            self.DIM = 3
        
    def __len__(self):
        return self.template.shape[0] 
    
    def __getitem__(self,index):
        if(self.Test):
            return torch.tensor(self.template[index,:,:self.DIM]).float(), \
                torch.tensor(self.source[index,:,:self.DIM]).float(), \
                torch.tensor(self.estimated_transform[index]).float() 
        else:
            return torch.tensor(self.template[index,:,:self.DIM]).float(),\
                torch.tensor(self.source[index,:,:self.DIM]).float(), \
                torch.tensor(self.transformation[index]).float() 
    
    def downsample_pcd(self, pcd_array):
        # :: WARNING :: 
        # :: Downsampling point cloud only implemented when 1 object in .hdf5 file, not for training sets
        
        # Turn point cloud array into point cloud
        r = 2 * self.voxel_size
        neigh_max = 30
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_array[0,:,0:3])
        
        # Downsample with given voxelsize
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_down_points = np.asarray(pcd_down.points)
        
        if(self.normals):
            pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=r, max_nn=neigh_max))
            
            pcd_normals = np.asarray(pcd_down.normals)
            pcd_down_points = np.concatenate((pcd_down_points,pcd_normals),1)
        
        pcd_down_array =  np.expand_dims(pcd_down_points, 0)
        return pcd_down_array


"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""

def remove_mean(pcd_array):
    # :: Remove mean from point cloud
    
    pcd_mean = np.mean(pcd_array,1)[:,0:3]
    # print(pcd_mean)
    pcd_array_no_mean = pcd_array
    pcd_array_no_mean[:,:,0:3] = pcd_array_no_mean[:,:,0:3] - pcd_mean
    return pcd_array_no_mean, pcd_mean

def remove_mean_transformation(transformation_array,mean):
    # :: Remove mean from translation vector in ground truth
        
    transformation_array_no_mean = transformation_array
    nmb_transf = transformation_array.shape[0]
    
    for i in range(nmb_transf):
        transformation_array_no_mean[i,0:3,3] = transformation_array_no_mean[i,0:3,3] - mean
    
    return transformation_array_no_mean