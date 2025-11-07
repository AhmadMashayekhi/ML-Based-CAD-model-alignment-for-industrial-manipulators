import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def frobeniusNormLoss(predicted, igt):
    """ 
    Computes difference between I and error, squares result and calculates 
    mean/sum over the 16 elements
    
    """
    """ |predicted*igt - I| (should be 0) """
    assert predicted.size(0) == igt.size(0)
    assert predicted.size(1) == igt.size(1) and predicted.size(1) == 4
    assert predicted.size(2) == igt.size(2) and predicted.size(2) == 4
    
    error = predicted.matmul(igt) #Error matrix for each element in the batch

    I = torch.eye(4).to(error).view(1, 4, 4).expand(error.size(0), 4, 4)
    
    #changed return to "reduction='sum'" instead of "size_average = True" and *16
    #changed return to "reduction='mean'"
    
    #Calculates square error between each pair of one element in batch (16 elements in tensor)
    #Afterwards take mean over 16 elements
    #Do same for other elements in batch and compute total mean over all elements in one batch
    
    return torch.nn.functional.mse_loss(error, I, reduction='mean')


class FrobeniusNormLoss(nn.Module):
    def __init__(self):
        super(FrobeniusNormLoss, self).__init__()

    def forward(self, predicted, igt):
        return frobeniusNormLoss(predicted, igt)