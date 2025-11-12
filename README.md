# PointNetLK_CAD_PCD_Alignment

This repository contains the full pipeline used for automatic CAD–PCD alignment of industrial robot bases using **PointNetLK** and **ICP refinement**.  
The workflow was implemented and validated as part of the XR-Twin project at Vrije Universiteit Brussel (VUB) / Flanders Make.

---

## 📁 Repository Structure

```
All important files/
├── Creating_random_poses.py       # Step 1: Generate random transformations for CAD models using PyBullet (json)
├── Creating_data_sets.py          # Step 2: Create synthetic training/testing datasets (HDF5)
├── main_train_code.py             # Step 3: Run main training loop (configures data & model)
  ├── train_PointNetLK.py          # Step 4: Train the registration model (PointNetLK)
├── Test_PointNetLK_ICP.py         # Step 5: Evaluate registration and visualize results
├── Input_Output/                  # Contains datasets, checkpoints, and results (ignored in Git)
└── src/                           # Source scripts and utilities
```

---

## ⚙️ Workflow Overview

### 1️⃣ Generate Random Meshes Poses
Use `Creating_random_poses.py` to randomly place robot meshes in 3D space; using forward kinematics of the robot.  
This script creates JSON files containing the randomly generated transformations .

### 2️⃣ Create Synthetic Datasets
Run `Creating_data_sets.py` to build paired datasets of CAD and PCD point clouds. All robot meshes, the CAD model and the ground transformation which relates them are stored in `.hdf5` format under:
```
Input_Output/datasets/
```

### 3️⃣ Configure and Launch Training
`main_train_code.py` defines the training configuration (epochs, learning rate, etc.).
It uses data prepared in Step 2 and calls the model definition from the Learning3D toolbox.

### 4️⃣ Train the PointNetLK Model
Run:
```bash
python train_PointNetLK.py
```
This script trains the PointNetLK model for rigid point cloud registration.
Checkpoints are saved to:
```
Input_Output/checkpoints/
```

### 5️⃣ Evaluate and Visualize Results
Finally, run:
```bash
python Test_PointNetLK_ICP.py
```
This script loads a trained model and performs registration followed by ICP refinement.  
It outputs JSON result files and optional 3D visualizations.

---

## 🧠 Dependencies

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- Open3D  
- NumPy  
- Matplotlib  
- tqdm  

You can install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## 🧩 Citation / Acknowledgment
If you use this work, please cite the following project:

**Mashayekhi, A.**, Denayer, M., Verstraten, T.  
_“Automatic CAD–PCD Alignment of Industrial Manipulator Bases Using PointNetLK and ICP Refinement”_,  
Vrije Universiteit Brussel (VUB), Flanders Make, 2025.

---

## 🧱 Notes

- Large files such as datasets and pretrained weights are ignored in this repository.  
- Reviewers can request the complete dataset and checkpoints separately.

---

📬 **Author:** Ahmad Mashayekhi  
Robotics & Multibody Mechanics (R&MM), Vrije Universiteit Brussel / Flanders Make  
ahmad.mashayekhi@vub.be
