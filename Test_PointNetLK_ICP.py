"""
Author: Ahmad Mashayekhi
Project: ML-Based CAD–PCD Alignment for Industrial Manipulators

Description
-----------
End-to-end driver script for running the CAD–PCD alignment experiment on a
selected robot base (Part_name). It:

  • Loads ground-truth JSON for the chosen robot from Input_Output/datasets/ground_truth
  • Runs a single PointNetLK registration experiment (run_one_experiment)
  • Visualizes the initial alignment result
  • Runs an ICP-based refinement step (run_refinement)
  • Visualizes the refined alignment result from Input_Output/results/PointNetLK_refined/refinement

Usage
-----
Set Part_name to one of:
    "Doosan_1", "Doosan_2", "Fanuc_1", "Fanuc_2", "Stabuli", "Universal_Robots"
then run:
    python main_run_experiment.py
"""


import os
import json
import numpy as np
import open3d as o3d
import copy
from pathlib import Path
from src.run_experiment import run_one_experiment, run_refinement, run_experiment_batch, run_refinement_batch
from src.visualizer import visualise_result, visualise_ground_truth
from src.ground_truth_accuracy import evaluate_ground_truth, evaluate_ground_truth_batch
from src.compute_errors import compute_errors, compute_errors_batch
from src.data_manager import preprocess_data


# Part_name = "Doosan_1"    
# Part_name = "Doosan_2"    
# Part_name = "Fanuc_1"    
# Part_name = "Fanuc_2"   
# Part_name = "Stabuli"    
Part_name = "Universal_Robots"    



"""
# =============================================================================
# -------------------------------RUN EXPERIMENT--------------------------------
# =============================================================================
# """
current_dir = os.path.dirname(os.path.abspath(__file__))
gt_json_file_exp = os.path.join(current_dir, "Input_Output", "datasets", "ground_truth", Part_name + ".json")

run_one_experiment(gt_json_file_exp, pcr_method = "PointNetLK", voxel_size = 0,
                    MSEThresh=0.00001, zero_mean = True)


# """
# =============================================================================
# ------------------------------VISUALIZE RESULT-------------------------------
# =============================================================================
# """
result_json_file = os.path.join(
    current_dir, "Input_Output", "results", "PointNetLK", "experiment", Part_name + "_result.json"
)

visualise_result(result_json_file)


"""
# =============================================================================
# -------------------------------REFINE RESULT---------------------------------
# =============================================================================
"""
# # print("------------------- step 1 -------------------")
# result_json_dir = BASE_DIR + "/results/PointNetLK/experiment/"
# result_json_file = result_json_dir + Part_name + "_result.json"


run_refinement(result_json_file, voxel_size = 0.1)

# print("------------------- step 2 -------------------")

refined_json_dir = os.path.join(
    current_dir, "Input_Output", "results", "PointNetLK_refined", "refinement"
)
refined_json_file = os.path.join(refined_json_dir, Part_name + "_result.json")


visualise_result(refined_json_file, voxel_size = 0)


