import numpy as np
import torch
#Metrics definition
epsilon=0.000000001


def dice_coef_metric(inputs, target):
    intersection = 2.0 * ((target * inputs).sum()) + epsilon
    union = target.sum() + inputs.sum() + epsilon
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return intersection / union


def accuracy(true_labels,predicted_labels):
    TP = np.sum(np.logical_and(true_labels == 1, predicted_labels == 1))
    TN = np.sum(np.logical_and(true_labels == 0, predicted_labels == 0))
    FP = np.sum(np.logical_and(true_labels == 1, predicted_labels == 0))
    FN = np.sum(np.logical_and(true_labels == 0, predicted_labels == 1))
    
    return (TP+TN)/(TP+TN+FP+FN)

#Segmentation Loss
def soft_dice_loss(inputs, target):
    epsilon = 0.00001
    intersection = 2.0 * ((target * inputs).sum()) + epsilon
    union = torch.sum(target**2) + torch.sum(inputs**2) + epsilon

    return 1 - (intersection / union)