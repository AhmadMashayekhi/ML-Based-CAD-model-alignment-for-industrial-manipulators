"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



data manager

Manage, save and process registration data

Inputs:
    - .json file
    
Output:
    - .json with results
"""


"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import os
import json
import torch
import numpy as np
import open3d as o3d
from pathlib import Path

"""
=============================================================================
---------------------------------VARIABLES-----------------------------------
=============================================================================
"""

# Project root is the parent of "src" (this file lives in src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input/Output data lives under Input_Output/datasets/...
IO_ROOT       = PROJECT_ROOT / "Input_Output"
DATASETS_ROOT = IO_ROOT / "datasets"

cad_path = DATASETS_ROOT / "CAD"
src_path = DATASETS_ROOT / "point_clouds"
gt_path  = DATASETS_ROOT / "ground_truth"

# Keep results at project root (unchanged location semantics)
# Place results inside Input_Output for consistency
out_path = IO_ROOT / "results"

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def read_json(path,key = None):
    """
    Read key of json file, located in path

    Parameters
    ----------
    path        : string            : path to .json file
    key         : string            : key of .json file to read
       
    Returns
    -------
    data        : data              : data loaded from .json file
    
    Link: https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/
    """
    path = Path(path)  # allow Path or str
    # Opening JSON file
    with open(path, 'r') as openfile:
 
        # Reading from json file
        json_object = json.load(openfile)
    
    if(key):
        return json_object[key]
    else:
        return json_object

def write_json(file_name, output_path, dictionary):
    output_path = Path(output_path)
    out_file = output_path / file_name

    json_object = json.dumps(dictionary, indent=3)

    output_path.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as outfile:
        outfile.write(json_object)

        
def save_result(file_name, experiment_name, estimated_transfo, 
                registration_time, dictionary, registration_parameters):

    # Location to save results
    out_folder = out_path / experiment_name               # use Path division
    out_file = file_name + "_result.json"

    # Data to be written
    dictionary["registration_parameters"] = registration_parameters
    dictionary["estimated_transformation"] = estimated_transfo.tolist()
    dictionary["frame"] = "template to source"            # removed trailing comma
    dictionary["result frame"] = "source to template"
    dictionary["registration time [s]"] = registration_time

    write_json(out_file, out_folder, dictionary)
    return



def prepare_stl(template_stl, nmb_source_points, multiple, scale = 1,
                Normals_radius = 0.01, Normals_Neighbours = 30):
    """
    Prepare .stl file to .ply file
    
    Parameters
    ----------
    template_stl        : Open3d TriangleMesh Object                        : .stl file of template
    nmb_source_points   : int                                               : number of points in source PC
    multiple            : float                                             : multiple of nmb_source_points
    scale               : float                                             : scale factor for template
    Normals_radius      : float (to estimate normal vectors on template)    : estimate normal vectors
    Normals_Neighbours  : float (to estimate normal vectors on template)    : estimate normal vectors
    
    Returns
    -------
    template_pointcloud : Open3D Point Cloud                                : template point cloud
    """
    
    nmb_template_points = nmb_source_points*multiple
    template_pointcloud = template_stl.sample_points_uniformly(number_of_points=nmb_template_points)  
    
    template_pointcloud.scale(1/scale,center=template_pointcloud.get_center())
    template_pointcloud.translate(-template_pointcloud.get_center())
    
    template_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=Normals_radius, max_nn=Normals_Neighbours))

    return template_pointcloud

def preprocess_data(json_file, voxel_size=None, zero_mean=False):
    """
    Load correct files from directory and preprocess.

    Parameters
    ----------
    json_file   : str | Path   : location of .json file
    voxel_size  : float | None : downsample point clouds if > 0
    zero_mean   : bool         : center point clouds

    Returns
    -------
    template_pointcloud : Open3D Point Cloud
    source_pointcloud   : Open3D Point Cloud
    transformation      : list (Nx4x4 or 4x4)
    json_info           : dict
    """
    from pathlib import Path

    # Ensure Path
    json_file = Path(json_file)

    # Read data
    json_info       = read_json(json_file)
    object_name     = json_info["name"]
    scan_nmb        = json_info["scan"]
    BB              = json_info["bounding_box"]
    sampling_ratio  = json_info["sampling_density"]
    scale           = json_info["scale"]

    # Allow JSON to override zero_mean/voxel_size if provided
    if "registration_parameters" in json_info:
        zero_mean = json_info["registration_parameters"].get("centered", zero_mean)
        if voxel_size is None:
            voxel_size = json_info["registration_parameters"].get("voxel_size", voxel_size)

    # Build PLY/STL paths
    if BB == 0:
        src_ply_dir = src_path / "filtered"
    else:
        src_ply_dir = src_path / "bounding_box" / f"BB_{BB}"

    src_base = f"{object_name}_{scan_nmb}"
    if BB != 0:
        src_base = f"{src_base}_BB_{BB}"

    src_ply_file = src_ply_dir / f"{src_base}_Source.ply"
    tmp_stl_file = cad_path / f"{object_name}.stl"

    # Fail fast if any input is missing
    missing = []
    if not src_ply_file.is_file():
        missing.append(f"PCD: {src_ply_file}")
    if not json_file.is_file():
        missing.append(f"JSON: {json_file}")
    if not tmp_stl_file.is_file():
        missing.append(f"CAD: {tmp_stl_file}")

    if missing:
        raise FileNotFoundError(
            "preprocess_data: required input(s) not found:\n  " + "\n  ".join(missing)
        )

    # Load data
    source_pointcloud = o3d.io.read_point_cloud(str(src_ply_file))
    template_stl      = o3d.io.read_triangle_mesh(str(tmp_stl_file))

    # Create template point cloud (sample STL)
    nmb_source_points   = len(np.asarray(source_pointcloud.points))
    template_pointcloud = prepare_stl(
        template_stl,
        nmb_source_points,
        sampling_ratio,
        scale
    )

    # Ground-truth transformation (template -> source)
    transformation = read_json(json_file, "transformation")

    # Optional centering
    if zero_mean:
        source_mean = source_pointcloud.get_center()
        source_pointcloud   = source_pointcloud.translate(-source_mean)
        template_pointcloud = template_pointcloud.translate(-template_pointcloud.get_center())
        transformation      = remove_mean_transformation(transformation, np.asarray(source_mean))

    # Optional downsampling
    if voxel_size and voxel_size > 0:
        source_pointcloud   = source_pointcloud.voxel_down_sample(voxel_size)
        template_pointcloud = template_pointcloud.voxel_down_sample(voxel_size)

    return template_pointcloud, source_pointcloud, transformation, json_info


def remove_mean_transformation(transformation,mean):
    """
    Remove mean from translation vector in ground truth
    
    Parameters
    ----------
    transformation                  : Nx4x4 list            : list of ground truth solutions
    mean                            : 1x3 numpy array       : computed mean of point cloud
    
    Returns
    ----------
    transformation_array_no_mean    : Nx4x4 list            : list of updated ground truth solutions
    """
    transformation_array = np.asarray(transformation)
    transformation_array_no_mean = transformation_array
    nmb_transf = transformation_array.shape[0]
    
    for i in range(nmb_transf):
        transformation_array_no_mean[i,0:3,3] = transformation_array_no_mean[i,0:3,3] - mean
    
    return transformation_array_no_mean.tolist()

def pointcloud_to_torch(pointcloud):
    """
    Turn point cloud object into torch tensor

    Parameters
    ----------
    pointcloud          : Open3D Point Cloud    : point cloud
       
    Returns
    -------
    tensor              : 1xNx6 Torch Tensor    : tensor
    """
    
    points_array = np.asarray(pointcloud.points)
    normals_array = np.asarray(pointcloud.normals)
    array = np.concatenate((points_array,normals_array),1)
    
    tensor = torch.tensor(array)
    tensor = tensor.expand(1,tensor.size(0),6)
    
    return tensor.float()

def invert_transformation(transformation):
    """
    Invert transformation matrix

    Parameters
    ----------
    transformation          : 4x4 list              : transformation matrix
       
    Returns
    -------
    inv_transformation      : 4x4 list              : inverted transformation matrix
    """
    
    transformation_matrix = np.asarray(transformation)
    inv_transformation = np.linalg.inv(transformation_matrix)
    return inv_transformation.tolist()