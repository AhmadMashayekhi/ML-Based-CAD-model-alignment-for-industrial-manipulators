"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



errors

Class to compute all the desired errors, comparing PCR performance.

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth
        o Estimated Transformation

Output:
    - Errors list
    
Credits:
    Chamfer distance & "get_transformations" function by vinit5, as part of the Learning3D toolbox
    LINK: https://github.com/vinits5/learning3d#use-your-own-data
    
    Creating tables using tabulate
    LINK: https://pypi.org/project/tabulate/
    
    Euler angles computation
    LINK: LINK: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
#General Imports
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

# For creating tables for the errors,
# LINK: https://pypi.org/project/tabulate/
from tabulate import tabulate 

# Chamfer Distance loss from learning3D toolbox, 
# LINK: https://github.com/vinits5/learning3d#use-your-own-data
from toolboxes.learning3d.losses import ChamferDistanceLoss
# from h5_files.file_reader import show_open3d


"""
=============================================================================
-----------------------------------DISPLAY-----------------------------------
=============================================================================
"""

def display_table(data, lim):
    names = [["Relative Error (Angular) (°):"],
             ["Relative Error (Translational) (mm):"],
             ["RMSE (Angular) (°):"], 
             ["RMSE (Translational) (mm):"],
             ["MAE (Angular) (°):"],
             ["MAE (Translational) (mm):"],
             ["Recall (%), threshold: " + str(lim) + ":"],
             ["Coefficient of Determination (R2):"],
             ["Chamfer Distance:"]]
    data = np.hstack([names,data])
    col_names = ["Error Name", "Value"]
    print("\n" + tabulate(data, headers=col_names,tablefmt="github"))
"""
=============================================================================
------------------------------------ERRORS-----------------------------------
=============================================================================
"""

def RelativeError(R_gt,t_gt,R_est,t_est_inv):
    """
    Error between estimated and ground truth rotation matrix and translation vector
    
    If more than one element in batch, compute mean over all elements in batch
    """
    
    #Angular error
    err = torch.bmm(R_gt,R_est)
    Error_angl = 0
    for i in range(err.size(0)):
        trace = float(torch.trace(err[i]))
        temp = (trace-1)/2
        if(abs(temp) <= 1):
            sign = math.asin(temp)/abs(math.asin(temp))
            Error_angl = Error_angl + sign*math.acos(temp) * 180/(math.pi)
        else:
            Error_angl = Error_angl
    Error_angl = Error_angl/err.size(0)
    
    #Translational error
    Error_trans = torch.mean(torch.norm(torch.add(t_est_inv.transpose(2,1),-t_gt.transpose(2,1)),dim=2))
    
    Errors = np.array([[Error_angl],
                       [float(Error_trans*1e3)]])
    
    return Errors

def RootMeanSquareError(R_gt,t_gt,R_est,t_est_inv,root =  True): #see Frobenius Norm for MSE on T
    """
    Error squared + mean over elements between estimated and ground truth 
    rotation matrix and translation vector
    """
    
    Error_R = torch.bmm(R_est,R_gt)
    I = torch.eye(3).to(Error_R).view(1, 3, 3).expand(Error_R.size(0), 3, 3)

    #Apply Mean Square Error function for all batches
    Error_angl = torch.nn.functional.mse_loss(Error_R, I, reduction='mean')
    Error_trans = torch.nn.functional.mse_loss(t_est_inv.transpose(2,1), t_gt.transpose(2,1), reduction='mean')
    
    if(root == True): #Taking root if asked
        Error_angl = math.sqrt(Error_angl)
        Error_trans = math.sqrt(Error_trans)
        
    Errors = np.array([[float(Error_angl)], 
                       [float(Error_trans*1e3)]])
    
    return Errors

def MeanAbsoluteError(R_gt,t_gt,R_est_inv,t_est_inv):
    """
    Mean of absolute value of the difference between estimated and ground truth
    translation vector and Euler angles
    """
    
    Error_angl = 0
    for i in range(R_gt.size(0)):
        Euler_gt = EulerAngles(R_gt[i][:][:])[0]
        Euler_est = EulerAngles(R_est_inv[i][:][:])[0]
        Error_angl = Error_angl + np.mean(np.abs(np.add(Euler_gt,-Euler_est)))*180/math.pi
    
    Error_angl = Error_angl/R_gt.size(0)
    Error_trans = torch.mean(torch.square(torch.add(t_est_inv.transpose(2,1),-t_gt.transpose(2,1))))
    
    Errors = np.array([[float(Error_angl)],
                       [float(Error_trans*1e3)]])
    
    return Errors

def Recall(R_gt,t_gt_inv,R_est_inv,t_est,source,tau): 
    """
    % of pairs for which RMSE < tau 
    """
    
    transfo_src = torch.add(torch.bmm(source,R_est_inv),t_est.transpose(2,1))
    tmpl = torch.add(torch.bmm(source,R_gt),t_gt_inv.transpose(2,1))
    
    # show_open3d(tmpl,transfo_src)
    
    Recall = 0
    
    for el in range(transfo_src.size(0)):
        for i in range(transfo_src.size(1)):
            MSE = torch.nn.functional.mse_loss(transfo_src[el][i][:],tmpl[el][i][:],reduction="mean")
            RMSE = math.sqrt(MSE)
            if(RMSE < tau):
                Recall = Recall + 1
    
    Recall = Recall/(transfo_src.size(0)*transfo_src.size(1))*100
    Errors = np.array([[Recall]])
    return Errors

#Coefficient of determination 

def Coeff_Determination(R_gt,t_gt_inv,R_est_inv,t_est,source):
    
    """
    %R2 = 1 - SSD/TSS
    uses R_gt & R_est_inv due to multiplication order in torch.bmm (transposed)
    """
    
    transfo_src = torch.add(torch.bmm(source,R_est_inv),t_est.transpose(2,1))
    tmpl = torch.add(torch.bmm(source,R_gt),t_gt_inv.transpose(2,1))

    mean = torch.mean(transfo_src,1)
    
    R2_tot = 0
    for el in range(transfo_src.size(0)):
        SSD = 0
        TSS = 0
        R2 = 0
        for i in range(transfo_src.size(1)):
            SSD = SSD + torch.square(torch.norm((transfo_src[el][i][:]-tmpl[el][i][:])))
            TSS = TSS + torch.square(torch.norm((transfo_src[el][i][:]-mean[el][:])))
        R2 = 1 - (SSD/TSS)
        R2_tot = R2_tot + R2
    R2_tot = R2_tot/3
    
    Errors = np.array([[float(R2)]])
    return Errors

def new_metric(transfo_src,template): 
    # -- unused --
    # :: mean of quadratic distance
    nmb_points = transfo_src.shape[1]
    #print(nmb_points)
    transfo_src = transfo_src[0].tolist()
    template = np.array(template[0])
    
    total_dist = 0
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(template)
    for i in tqdm(range(nmb_points)):
        point = np.expand_dims(transfo_src[i][:3],0)
        #print(point)
        distc, idx = nbrs.kneighbors(point)
        total_dist = total_dist + float(distc[0])*float(distc[0])
        #print(idx)
    metric = total_dist/nmb_points
    return metric

"""
=============================================================================
------------------------------------OTHER------------------------------------
=============================================================================
"""

def get_transformations(tensor): 
    R_ba = tensor[:, 0:3, 0:3]                          # Ps = R_ba * Pt, should be =~ output['est_R']
    translation_ba = tensor[:, 0:3, 3].unsqueeze(2)     # Ps = Pt + t_ba
    R_ab = R_ba.permute(0, 2, 1)                        # Pt = R_ab * Ps, inverse/transposed of R_ba
    translation_ab = -torch.bmm(R_ab, translation_ba)   # Pt = Ps + t_ab
    
    temp = torch.concat([R_ab,translation_ab],axis=2)
    row = torch.tensor([[[0,0,0,1]]]).to(temp).expand(temp.size(0),1,4)
    tensor_inv = torch.concat([temp,row],axis=1) #Pt = igt_inv * Ps
    return  R_ba, translation_ba , R_ab, translation_ab, tensor_inv

def EulerAngles(R): 
    # :: Computes Euler angles from rotation matrix
    # LINK: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    
    if(R[2,0] != 1 or -1):
        theta_1 = -math.cos(R[2,0])
        theta_2 = math.pi-theta_1
        
        psi_1 = math.atan2(R[2,1]/math.cos(theta_1),R[2,2]/math.cos(theta_1))
        psi_2 = math.atan2(R[2,1]/math.cos(theta_2),R[2,2]/math.cos(theta_2))
        
        phi_1 = math.atan2(R[2,1]/math.cos(theta_1),R[0,0]/math.cos(theta_1))
        phi_2 = math.atan2(R[2,1]/math.cos(theta_2),R[0,0]/math.cos(theta_2))
    else:
        phi_1 = 0
        phi_2 = math.pi
        if(R[2,0] == -1):
            theta_1 = math.pi/2
            theta_2 = -3*math.pi/2
            psi_1 = math.atan2(R[0,1],R[0,2])
            psi_2 = math.pi + math.atan2(R[0,1],R[0,2])
        else:
            theta_1 = -math.pi/2
            theta_2 = 2*math.pi/2
            psi_1 = math.atan2(-R[0,1],-R[0,2])
            psi_2 = -math.pi + math.atan2(-R[0,1],-R[0,2])
    return np.array([[theta_1,psi_1,phi_1],[theta_2,psi_2,phi_2]])

"""
=============================================================================
-------------------------------INITIALIZATION--------------------------------
=============================================================================
"""

class Errors(nn.Module):
    def __init__(self,template,source,igt,T_est,recall_lim=0.01):
        super(Errors, self).__init__()
        self.template = template
        self.source = source
        # self.transformed_source = output['transformed_source']
        self.igt = igt
        self.T_est = T_est
        self.recall_lim = recall_lim
        
    def forward(self):
        R_gt, t_gt, R_gt_inv, t_gt_inv, gt_inv = get_transformations(self.igt)
        R_est, t_est, R_est_inv, t_est_inv, T_est_inv = get_transformations(self.T_est)
        
        #compute errors
        self.RelErr = RelativeError(R_gt,t_gt,R_est,t_est_inv)
        self.RMSE = RootMeanSquareError(R_gt,t_gt,R_est,t_est_inv)
        self.MAE = MeanAbsoluteError(R_gt,t_gt,R_est_inv,t_est_inv)
        self.Recall = Recall(R_gt,t_gt_inv,R_est_inv,t_est,self.source, self.recall_lim)
        self.R2 = Coeff_Determination(R_gt,t_gt_inv,R_est_inv,t_est,self.source)

        CDLoss = ChamferDistanceLoss()(self.T_est, self.igt).detach().cpu().numpy()
        self.CD = np.array([[CDLoss]])
        
        data = (self.RelErr,self.RMSE,self.MAE,self.Recall,self.R2,self.CD)
        data = np.vstack(data)
        return data
    
    def display(self, data):
        display_table(data, self.recall_lim)