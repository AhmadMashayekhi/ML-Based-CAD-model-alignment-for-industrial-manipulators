""""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



h5 writer

Various functions allowing to create .hdf5 files in various formats. Functions
include:
    - write_h5 
        o file containing template, source, gt
    - write_h5_result
        o file containing template, source, gt, estimated transformation
    - append_h5
        o append new data to given h5 file
    - write_h5_sep 
        o R/t seperated
    - write_h5_labeled 
        o h5 file with labels of objects
    - create_DIR
        o creates directory based on folder & file name
    - uniquify
        o 

Inputs:
    - .ply file (Source, RealSense)

Output:
    - .txt file containing:
        o Source (w/ normals)

Credits:
    Uniquify function
    LINK: https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    
    appending data to .hdf5 file
    LINK: https://stackoverflow.com/a/47074545
    
    Farthest Subsample Point function by vinits5 as part of the Learning3D library 
    (learning3d: dataloaders.py)
    Link: https://github.com/vinits5/learning3d#use-your-own-data

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import h5py
import numpy as np
import torch
import math
import open3d as o3d
import copy
import os

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

BASE_DIR = os.getcwd() #Parent folder -> Thesis
# print(BASE_DIR)
import h5_files.file_reader as r
# from misc.transformer import camera_model2

"""
=============================================================================
------------------------------SAVING & READING-------------------------------
=============================================================================
"""
def write_h5(File_Name,template, source, gt, FolderName = ""): #Create h5 file in correct configuration
    BASE_DIR = create_DIR(File_Name, FolderName)

    batch_size = template.shape[0]

    if(gt.all() == None):
        gt_ = np.array(size=(batch_size,4,4))
    else:
        gt_ = gt

    with h5py.File(BASE_DIR,"w") as data_file:
        data_file.create_dataset("template",data=template)
        data_file.create_dataset("source",data=source)
        data_file.create_dataset("transformation",data=gt_)
        data_file.create_dataset("transformation_all",data=gt_,chunks=True, maxshape=(None,4,4))
        
    print("\n:: File saved as: ", BASE_DIR)
    return

def write_h5_result(file_loc, method, voxel_size, Test, FolderName = "results"): 
    #Create h5 file in correct configuration for result
    
    template, source, gt_, gt_symm = r.h5file_to_torch(file_loc,zero_mean=False)
    
    File_Name = os.path.basename(file_loc).rsplit('.hdf5')[0] # Extract file name from path
    if(voxel_size != 0):
        File_Name = File_Name + "_vs_" + str(voxel_size)
    File_Name = File_Name + "_" + method
    BASE_DIR = create_DIR(File_Name, FolderName)

    with h5py.File(BASE_DIR,"w") as data_file:
        data_file.create_dataset("template",data=template)
        data_file.create_dataset("source",data=source)
        data_file.create_dataset("transformation",data=gt_)
        data_file.create_dataset("transformation_all",data=gt_symm)
        data_file.create_dataset("Test",(0,4,4),chunks=True, maxshape=(None,4,4))
        
    print("\n:: File saved as: ", BASE_DIR)
    return BASE_DIR

def write_h5_sep(File_Name,template, source, Rab, tab, FolderName = ""): #Create h5 file in correct configuration
    BASE_DIR = create_DIR(File_Name, FolderName)

    with h5py.File(BASE_DIR,"w") as data_file:
        data_file.create_dataset("template",data=template)
        data_file.create_dataset("source",data=source)
        data_file.create_dataset("rotation",data=Rab)
        data_file.create_dataset("translation",data=tab)
    return

def append_h5(file_loc,key,new_data):
    # Adds data to an existing key in the given h5 file
    # Appending arrays to h5 file, source: https://stackoverflow.com/a/47074545
    
    " Read h5 file content "
    h5file = h5py.File(file_loc, 'a')      # open the file
    data = h5file[key]                     # load the data
    # print(data)
    
    " Increase size to add new matrix "
    h5file[key].resize(data.shape[0]+1,axis=0)
    # print(data)
    
    data[data.shape[0]-1,:,:] = new_data    # assign new values to data
    # print(data)
    h5file.close()                          # close the file

def write_h5_labeled(File_Name,pointcloud,labels, FolderName):
    # :: Save labeled point clouds
    
    BASE_DIR = create_DIR(File_Name, FolderName)

    with h5py.File(BASE_DIR,"w") as data_file:
        data_file.create_dataset("pcs",data=pointcloud)
        data_file.create_dataset("labels",data=labels)
    return

def create_DIR(File_Name, FolderName):
    # :: Create directory for given file name & folder name to save .hdf5 file results
    
    if(FolderName == ""):
        FolderName = File_Name
    DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/"

    if not os.path.exists(DIR + FolderName):
        os.makedirs(DIR + FolderName)
    DIR = DIR + FolderName + "/"
    DIR = DIR + File_Name + ".hdf5"
    DIR = uniquify(DIR)
    return DIR

def uniquify(path):
    # :: Check if path exists and if so add number
    # :: LINK: https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1

    return path

def save_as_txt(pcd,name):
    # :: Save point cloud as .txt file
    
    print(":: Saving file...")
    save = np.asarray(pcd.points)

    DIR = os.path.join(os.getcwd() + "/h5_files/output/processing_results/" + name + ".txt")
    DIR = uniquify(DIR)
    np.savetxt(DIR, save)

    print(":: Saving finished, saved as: ", DIR)
    return

"""
=============================================================================
-------------------------------TRANSFORMING PC-------------------------------
=============================================================================
"""

def add_floor(pointcloud):
    # -- unused --
    corners = box_cloud(pointcloud)

    #minx = np.min(corners[:,0])
    miny = np.min(corners[:,1])
    #minz = np.min(corners[:,2])

    index = 1

    rect_corners = []
    for i in range(8):
        if(corners[i][index]==miny):
            rect_corners.append(corners[i].tolist())

    plane_points = sample_plane(np.array(rect_corners))

    new_pointcloud = pointcloud[0].tolist() + plane_points

    new_pointcloud = torch.tensor(new_pointcloud)
    new_pointcloud = new_pointcloud.expand(1,new_pointcloud.size(0),3)

    return new_pointcloud

def sample_plane(corners):
    # -- unused --
    points_list = []

    diff = corners[2]-corners[0]
    dist = np.linalg.norm(diff)
    delta = 0.1

    step = delta*diff/dist

    nmb_points = int(dist/np.linalg.norm(step))

    first_line = sample_line(corners[0], corners[1])
    new_line = first_line
    points_list = first_line
    #print(np.array(points_list).shape)
    for i in range(nmb_points):
        new_line = new_line + step
        points_list = points_list + new_line.tolist()
        #print(np.array(points_list).shape)
    return points_list

def sample_line(begin,end):
    # -- unused --
    points_list = []

    diff = end-begin
    dist = np.linalg.norm(diff)
    delta = 0.1
    step = delta*diff/dist
    nmb_points = int(dist/np.linalg.norm(step))

    new_point = begin
    points_list.append(new_point.tolist())
    for i in range(nmb_points):
        new_point = new_point + step
        points_list.append(new_point.tolist())
    return points_list

"""
-------------------FROM learning3d: dataloaders.py-------------------
"""
def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
    return pointcloud1[idx1, :], gt_mask

"""
---------------------------------------------------------------------
"""

def apply_transfo(template,R,t,normals=False):
    # :: Apply rigid transformation to template 
    
    if(normals == False):
        source = np.zeros((1,template.shape[1],3))
        for i in range(template.shape[1]):
            #print(template[0][i][:])
            #print(np.dot(template[0][i][:],R))
            source[0][i][:] = np.add(np.matmul(R,template[0][i][:]),t)
    else:
        source = np.zeros((1,template.shape[1],6))
        for i in range(template.shape[1]):
            #print(template[0][i][:])
            #print(np.dot(template[0][i][:],R))
            source[0][i][0:3] = np.add(np.matmul(R,template[0][i][0:3]),t)
            source[0][i][3:6] = np.matmul(R,template[0][i][3:6])            #Normals only rotated
    return source

def rotation_matrix(angle,axis):
    # :: Create rotation matrix with angle over axis
    
    angle = angle*math.pi/180
    R = np.eye(3)
    if(axis == "z"):
        R[0,:] = [math.cos(angle), math.sin(angle), 0]
        R[1,:] = [-math.sin(angle), math.cos(angle), 0]
    elif(axis == "y"):
        R[0,:] = [math.cos(angle), 0, math.sin(angle)]
        R[2,:] = [-math.sin(angle),0, math.cos(angle)]
    else:
        R[1,:] = [0, math.cos(angle), -math.sin(angle)]
        R[2,:] = [0, math.sin(angle), math.cos(angle)]
    return R

def random_rotation():
    # :: Create random rotations over random axis in [0,45] degrees
    
    theta1 = float(np.random.rand(1)*45)
    theta2 = float(np.random.rand(1)*45)
    theta3 = float(np.random.rand(1)*45)

    R1 = rotation_matrix(theta1, "x")
    R2 = rotation_matrix(theta2, "y")
    R3 = rotation_matrix(theta3, "z")

    R_ = np.matmul(R1,R2)
    R = np.matmul(R_,R3)

    return R

def homogenous_transfo(R,t): 
    # :: Create Ground Truth from rotation matrix & translation vector
    T = np.zeros((4,4))
    T[3,3] = 1
    T[0:3,0:3] = R
    T[:3,3] = t
    return T

def add_Gaussian_Noise(mu,sigma,orig_cloud,bound):
    # :: Add Gaussian Noise from a normal distribution with mean mu and deviation sigma,
    # :: clipped to [-bound,+bound] to the given array of points ([1,N,3]).
    noisy_cloud = copy.deepcopy(orig_cloud);
    for i in range(noisy_cloud.shape[1]):
        noise = np.clip(np.random.normal(0,sigma,(1,3)),-bound,bound)
        noisy_cloud[0][i][0:3] = np.add(noisy_cloud[0][i][0:3],noise)
    return noisy_cloud

def box_cloud(point_cloud):
    # :: Determine point cloud bounding box
    template_ = o3d.geometry.PointCloud()                               #Create point cloud
    template_.points = o3d.utility.Vector3dVector(point_cloud[0][:][:]) #Add points to cloud
    aabb = template_.get_axis_aligned_bounding_box()                    #Extract bounding box

    corners = np.asarray(aabb.get_box_points())
    return corners


"""
-------------------FROM PointNetLK (unused)-------------------
"""

class Resampler:
    """ [N, D] -> [M, D] """
    def __init__(self, num):
        self.num = num

    def __call__(self, tensor):
        num_points, dim_p = tensor.size()
        out = torch.zeros(self.num, dim_p).to(tensor)

        selected = 0
        while selected < self.num:
            remainder = self.num - selected
            idx = torch.randperm(num_points)
            sel = min(remainder, num_points)
            val = tensor[idx[:sel]]
            out[selected:(selected + sel)] = val
            selected += sel
        return out

"""
=============================================================================
----------------------------EXAMPLES (for info)------------------------------
=============================================================================
"""
    
def main():
    """ ----------------------------- """
    """ Example 01: Write Own h5-file """
    """ ----------------------------- """

    filename = BASE_DIR + "/toolboxes/learning3d/data/modelnet40_ply_hdf5_2048/ply_data_test0.h5"

    Dataset = r.h5reader(filename,"data")           # Shape: B x N x 3 (B = Batch-size, N = #points)
    #print(Dataset.shape)

    template = Dataset[1:2]                         # Shape: 1 x N x 3 (N = #points)
    #template = template + [1,1,1]
    #print(template.shape)

    t = [0,0,0.03]                                  # Translation vector, Ps = Pt + t
    #R = rotation_matrix(30,"y")                    # Rotation matrix,    Ps = R*Pt
    R = random_rotation()

    source = apply_transfo(template,R, t)           # Apply transformation
    #source = source + [1,1,1]                      # Move to simulate camera
    #source = camera_model2(source,[1,1,1],[0,0,0]) # Remove points to simulate camera
    #source = add_Gaussian_Noise(0, 0.03, source)   # Add noise with standard deviation

    gt = homogenous_transfo(R,t)                    # Homogenous transformation constructed from R and t

    #write_h5("test_h5file",template,source,gt)     # Create h5 file with template, source, ground truth

    #r.show_open3d(template, source)                # Show point cloud in Open3d

    """ -----------------------------------------------"""
    """ Example 02: Extract Bounding Box from Template """
    """ -----------------------------------------------"""

    box_cloud_corners = box_cloud(template)

    """ ----------------------------------------------------------------------------------------"""
    """ Example 03: Read Off-File and turn to Point Cloud (h5-file) (EXAMPLE 04 BETTER RESULTS) """
    """ ----------------------------------------------------------------------------------------"""

    #PointNetLK from GitHub needed xx deleted xx

    # BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
    # filename = BASE_DIR + "/datasets/ModelNet40/car/test/car_0198.off"
    # Off_file = r.offread(filename,True)                                     # Read off-file (PointNetLK)
    # v = Off_file.vertex_array                                               # Turn mesh into vertex array
    # tensor_ = torch.from_numpy(v).type(dtype=torch.float)                   # Turn vertex array into tensor
    # #print(tensor_.shape)

    # device = o3d.core.Device("CPU:0")                                       # Device type (CPU/GPU)
    # dtype = o3d.core.float32                                                # Points type

    # pcd = o3d.t.geometry.PointCloud(device)                                 # Assign device

    # pcd.point["positions"] = o3d.core.Tensor(tensor_.numpy(), dtype, device)# Assign points to point cloud

    # mesh = o3d.io.read_triangle_mesh(filename)                              # Create mesh

    # #o3d.visualization.draw([pcd,mesh])                                     # Visualize PC and Mesh

    # resampler = Resampler(1024)
    # template = resampler(tensor_)
    # template = template.expand(1,template.size(0),3)
    # source = apply_transfo(template,R, t)

    # write_h5("test_Off",template,source,gt)                                 # Create h5 file
    # #h5.show_open3d(template, source)                                       # Show point cloud in Open3d

    """ -----------------------------------------------------------------------------"""
    """ Example 04: Read Mesh-file (.obj/.stl/...) and turn to Point Cloud (h5-file) """
    """ -----------------------------------------------------------------------------"""

    #File can be .obj/.stl/.ply/.off/.gltf/.glb
    filename = BASE_DIR + "/datasets/CAD/Original/Base-Top_Plate.stl"

    mesh = o3d.io.read_triangle_mesh(filename)                  # Read Mesh file
    # mesh.compute_vertex_normals()
    # print(np.asarray(mesh.triangle_normals))
    # o3d.visualization.draw_geometries([mesh])

    pcd = mesh.sample_points_uniformly(number_of_points=4096)   # Sample Mesh uniformly
    #o3d.visualization.draw_geometries([pcd,mesh])              # Visualize PC and Mesh
    pcd.paint_uniform_color([0.11372549, 0.14509804, 0.87843137])
    # o3d.visualization.draw_geometries([pcd])

    template = torch.from_numpy(np.asarray(pcd.points))         # Turn PC to tensor of points
    template = template.expand(1,template.size(0),3)            # Correct dimensions, 1xNx3
    #print(template.shape)
    
    # Compute normal information
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries([mesh])
    
    # o3d.visualization.draw_geometries([pcd])

    mean = torch.mean(template,1)                               # Calculate PC mean
    template = template - mean                                  # Remove mean from PC to center
                                                                # on (0,0,0)

    t = [0,0,1]                                                 # Translation vector, Ps = Pt + t
    R = rotation_matrix(40,"z")                                 # Rotation matrix,    Ps = R*Pt
    gt = homogenous_transfo(R,t)
    source = apply_transfo(template,R, t)
    # source = add_Gaussian_Noise(0, 0.3, source, 1)
    
    # write_h5("test_STL",template,source,gt)                    # Create h5 file
    # r.show_open3d(template, source)                            # Show point cloud in Open3d

    """ ---------------------------------------------------"""
    """ Example 05: Create h5-file with normal information """
    """ ---------------------------------------------------"""

    filename = BASE_DIR + "/toolboxes/learning3d/data/modelnet40_ply_hdf5_2048/ply_data_test0.h5"
    
    Dataset = r.h5reader(filename,"data")                     # Shape: B x N x 3 (B = Batch-size, N = #points)
    #print(Dataset.shape)
    Normals = r.h5reader(filename, "normal")                  # Shape: B x N x 3
    #print(Normals.shape)

    Dataset = np.append(Dataset,Normals,2)                    # Shape: B x N x 6 (added normals)
    #print(Dataset.shape)

    template = Dataset[1:2]                                   # Shape: 1 x N x 6 (N = #points)
    #print(template.shape)

    t = [0,0,1]                                               # Translation vector, Ps = Pt + t
    R = rotation_matrix(10,"y")                               # Rotation matrix,    Ps = R*Pt

    source = apply_transfo(template,R, t, normals = True)
    #source = add_Gaussian_Noise(0, 0.03, source)             # Add noise with standard deviation

    gt = homogenous_transfo(R,t)                              # Homogenous transformation constructed from R and t
    #print(gt)

    #write_h5("test_normals",template,source,gt)              # Create h5 file with template, source, ground truth
    #r.show_open3d(template[:,:,0:3], source[:,:,0:3])        # Show point cloud in Open3d

    """ ---------------------------------------"""
    """ Example 06: Load source from text file """
    """ ---------------------------------------"""
    
    filename = BASE_DIR + "/h5_files/output/processing_results/Old/realsense_Square-Peg1.txt"
    source_ = r.txt_to_array(filename)
    
    # cam_dir = np.mean(source_,0)

    source = torch.from_numpy(source_)
    source = source.expand(1,source.size(0),3)
    # print(source.shape)

    filename = BASE_DIR + "/datasets/CAD/Original/Base-Top_Plate.stl"
    mesh = o3d.io.read_triangle_mesh(filename)                              # Read Mesh file

    pcd = mesh.sample_points_uniformly(number_of_points=source.shape[1]*2)  # Sample Mesh uniformly
    #o3d.visualization.draw_geometries([pcd,mesh])                          # Visualize PC and Mesh

    template = torch.from_numpy(np.asarray(pcd.points)/100)                 # Turn PC to tensor of points
    #print(template.shape)

    mean = torch.mean(template,0)                                           # Calculate PC mean
    template = template - mean                                              # Remove mean from PC to center
                                                                            # on (0,0,0)
    #template = template + cam_dir                                          # Place PC at mean of Source

    #template = camera_model(template,cam_dir)                              # Remove pionts from POV camera
    template = template.expand(1,template.size(0),3)
    mean = torch.mean(source,  1)                                           # Calculate PC mean
    source = source - mean                                                  # Remove mean from PC to center
                                                                            # on (0,0,0)

    t = [0,0,1]                                                             # Translation vector, Ps = Pt + t
    R = rotation_matrix(10,"y")                                             # Rotation matrix,    Ps = R*Pt
    gt = homogenous_transfo(R,t)

    #r.show_open3d(template[:,:,0:3], source[:,:,0:3])
    #write_h5("test_scan",template,source,gt)

    """ --------------------------------------"""
    """ Example 07: Add floor to point cloud  """
    """ --------------------------------------"""

    source = torch.from_numpy(source_)
    source = source.expand(1,source.size(0),3)

    mean = torch.mean(source,  1)                                         # Calculate PC mean
    source = source - mean                                                # Remove mean from PC to center
                                                                          # on (0,0,0)

    #print(source.shape)

    filename = BASE_DIR + "/datasets/CAD/Original/Square-Peg.stl"
    mesh = o3d.io.read_triangle_mesh(filename)                              # Read Mesh file

    pcd = mesh.sample_points_uniformly(number_of_points=source.shape[1]*2)  # Sample Mesh uniformly
    # o3d.visualization.draw_geometries([pcd])                          # Visualize PC and Mesh

    template = torch.from_numpy(np.asarray(pcd.points)/100)                 # Turn PC to tensor of points
    #print(template.shape)

    mean = torch.mean(template,0)                                           # Calculate PC mean
    template = template - mean                                              # Remove mean from PC to center
    #                                                                       # on (0,0,0)
    #template = template + cam_dir                                          # Place PC at mean of Source

    #template = camera_model(template,cam_dir)                              # Remove pionts from POV camera
    template = template.expand(1,template.size(0),3)
    #template = add_floor(template)
    #Alternative: Inventor

    t = [0,0,1]                                                             # translation vector, Ps = Pt + t
    R = rotation_matrix(10,"y")                                             # rotation matrix,    Ps = R*Pt
    gt = homogenous_transfo(R,t)

    # r.show_open3d(template[:,:,0:3], source[:,:,0:3])
    #write_h5("test_floor",template,source,gt)
    
    """ ----------------------------------------------------"""
    """ Example 09: Create 2 txt files for Source/Template  """
    """ ----------------------------------------------------"""
    
    filename = BASE_DIR + "/h5_files/output/processing_results/Old/realsense_Base-Top_Plate1.txt"
    source_ = r.txt_to_array(filename)
    
    # cam_dir = np.mean(source_,0)

    source = torch.from_numpy(source_)
    source = source.expand(1,source.size(0),3)
    # print(source.shape)

    filename = BASE_DIR + "/datasets/CAD/Original/Base-Top_Plate.stl"
    mesh = o3d.io.read_triangle_mesh(filename)                              # Read Mesh file

    pcd = mesh.sample_points_uniformly(number_of_points=source.shape[1]*2)  # Sample Mesh uniformly
    #o3d.visualization.draw_geometries([pcd,mesh])                          # Visualize PC and Mesh

    template = torch.from_numpy(np.asarray(pcd.points)/100)                 # Turn PC to tensor of points
    #print(template.shape)

    mean = torch.mean(template,0)                                           # Calculate PC mean
    template = template - mean                                              # Remove mean from PC to center
                                                                            # on (0,0,0)
    #template = template + cam_dir                                          # Place PC at mean of Source

    #template = camera_model(template,cam_dir)                              # Remove pionts from POV camera
    template = template.expand(1,template.size(0),3)
    mean = torch.mean(source,  1)                                           # Calculate PC mean
    source = source - mean                                                  # Remove mean from PC to center
                                                                            # on (0,0,0)
                                                                            
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template[0][:][:])
    source_.points = o3d.utility.Vector3dVector(source[0][:][:])
    template_.paint_uniform_color([0, 0, 1])
    source_.paint_uniform_color([1, 0, 0])

    # o3d.visualization.draw_geometries([template_])
    # o3d.visualization.draw_geometries([source_])

    # save_as_txt(template_, "Base-Top_Plate_template")
    # save_as_txt(source_, "Base-Top_Plate_source")
    
    
    # transformed_source = o3d.io.read_point_cloud(BASE_DIR + "/_test/GO-ICP/Base-Top_Plate.ply")
    # o3d.visualization.draw_geometries([template_,transformed_source])
    
    """ ------------------------------------------------------"""
    """ Example 10: Load source from text file (with normals) """
    """ ------------------------------------------------------"""
    
    filename = BASE_DIR + "/h5_files/output/processing_results/Old/Base-Top_Plate_Normals.txt"
    source_ = r.txt_to_array(filename)
    
    # cam_dir = np.mean(source_,0)

    source = torch.from_numpy(source_)
    source = source.expand(1,source.size(0),6)
    # print(source.shape)

    filename = BASE_DIR + "/datasets/CAD/Original/Base-Top_Plate.stl"
    mesh = o3d.io.read_triangle_mesh(filename)                              # Read Mesh file

    pcd = mesh.sample_points_uniformly(number_of_points=source.shape[1])  # Sample Mesh uniformly
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd,mesh])                          # Visualize PC and Mesh
    
    points = np.asarray(pcd.points)/100
    normals = np.asarray(pcd.normals)
    template = torch.from_numpy(np.concatenate((points,normals),1))                 # Turn PC to tensor of points
    #print(template.shape)

    mean = torch.mean(template,0)                                           # Calculate PC mean
    template = template - mean                                              # Remove mean from PC to center
                                                                            # on (0,0,0)
    #template = template + cam_dir                                          # Place PC at mean of Source

    #template = camera_model(template,cam_dir)                              # Remove pionts from POV camera
    template = template.expand(1,template.size(0),6)
    mean = torch.mean(source, 1)                                            # Calculate PC mean
    source = source - mean                                                  # Remove mean from PC to center
                                                                            # on (0,0,0)

    t = [0,0,1]                                                             # Translation vector, Ps = Pt + t
    R = rotation_matrix(10,"y")                                             # Rotation matrix,    Ps = R*Pt
    gt = homogenous_transfo(R,t)

    # r.show_open3d(template[:,:,0:3], source[:,:,0:3])
    # write_h5("test_scan_normals",template,source,gt)
    
    """ ------------------------------------------------------------"""
    """ Example 11: Show difference in real scale vs training scale """
    """ ------------------------------------------------------------"""
    
    filename = BASE_DIR + "/datasets/CAD/Original/Base-Top_Plate.stl"
    [Train_PC,_,Train_PC_max] = r.read_stl(filename,Nmb_points=10240,Scale=True)
    print(Train_PC_max)
    [Orign_PC,_,_] = r.read_stl(filename,Nmb_points=1024,Scale=False)
    Orign_PC = Orign_PC/100;
    
    Train_PC_tensor = torch.from_numpy(Train_PC);
    Train_PC_tensor = Train_PC_tensor.expand(1,10240,3)
    Orign_PC_tensor = torch.from_numpy(Orign_PC);
    Orign_PC_tensor = Orign_PC_tensor.expand(1,1024,3)

    t = [1,0,0]                                                             # Translation vector, Ps = Pt + t
    R = rotation_matrix(0,"y")                                             # Rotation matrix,    Ps = R*Pt
    
    Orign_PC_tensor = apply_transfo(Orign_PC_tensor,R, t, normals = False)

    # r.show_open3d(Train_PC_tensor, Orign_PC_tensor)
    
    """ --------------------------------------------------"""
    """ Example 12: Show different types of training data """
    """ --------------------------------------------------"""
    
    nmb_points = 10240
    filename = BASE_DIR + "/datasets/CAD/Original/Base-Top_Plate.stl"
    filename_floor = BASE_DIR + "/datasets/CAD/Floor/Base-Top_Plate.stl"
    object_mesh = o3d.io.read_triangle_mesh(filename)                              # Read Mesh file
    [Orign_PC,_,_] = r.read_stl(filename,Nmb_points=nmb_points,Scale=False, Center = False)
    [Floor_PC,_,_] = r.read_stl(filename_floor,Nmb_points=nmb_points,Scale=False, Center = False)
    
    Orign_PC_tensor = torch.from_numpy(Orign_PC);
    Floor_PC_tensor = torch.from_numpy(Floor_PC);
    Orign_PC_tensor_ext = Orign_PC_tensor.expand(1,nmb_points,3)
    
    Noisy_PC_tensor = add_Gaussian_Noise(0, 0.1, Orign_PC_tensor_ext,0.5) 
    
    Partial_PC_tensor,_ = farthest_subsample_points(Orign_PC_tensor,int(0.5*nmb_points))

    Floor_Partial_tensor,_ = farthest_subsample_points(Floor_PC_tensor,int(0.5*nmb_points))
    Floor_Partial_tensor = Floor_Partial_tensor.expand(1,int(0.5*nmb_points),3)
    Floor_Partial_Noisy_tensor = add_Gaussian_Noise(0, 0.1, Floor_Partial_tensor, 0.5)

    Original_PC = o3d.geometry.PointCloud()
    Noisy_PC = o3d.geometry.PointCloud()
    Partial_PC = o3d.geometry.PointCloud()
    Floor_PC = o3d.geometry.PointCloud()
    
    Original_PC.points = o3d.utility.Vector3dVector(Orign_PC_tensor)
    Noisy_PC.points = o3d.utility.Vector3dVector(Noisy_PC_tensor[0][:][:])
    Partial_PC.points = o3d.utility.Vector3dVector(Partial_PC_tensor)
    Floor_PC.points = o3d.utility.Vector3dVector(Floor_Partial_Noisy_tensor[0][:][:])
    
    Original_PC.paint_uniform_color([0, 0, 1])
    Noisy_PC.paint_uniform_color([0, 0, 1])
    Partial_PC.paint_uniform_color([0, 0, 1])
    Floor_PC.paint_uniform_color([0,0,1])
    
    o3d.visualization.draw_geometries([object_mesh,Original_PC])
    o3d.visualization.draw_geometries([object_mesh,Noisy_PC])
    o3d.visualization.draw_geometries([object_mesh,Partial_PC])
    o3d.visualization.draw_geometries([object_mesh,Floor_PC])
    
if __name__ == '__main__':
    main()