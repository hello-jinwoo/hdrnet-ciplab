# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:23:49 2018
@author: phamh
"""
import cv2
import numpy as np

# TODO: Integrate

def compute_angular_error(y_true, y_pred):
    """
    Angle between the RGB triplet of the measured ground truth
    illumination and RGB triplet of estimated illuminant

    Args:
        y_true (np.array): ground truth RGB illuminants
        y_pred (np.array): predicted RGB illuminants
    Returns:
        err (np.array):  angular error
    """

    gt_norm = np.linalg.norm(y_true, axis=1)
    gt_normalized = y_true / gt_norm[..., np.newaxis]
    est_norm = np.linalg.norm(y_pred, axis=1)
    est_normalized = y_pred / est_norm[..., np.newaxis]
    dot = np.sum(gt_normalized * est_normalized, axis=1)
    err = np.degrees(np.arccos(dot))
    return err

def compute_angular_error_stats(ang_err):
    """
    Angular error statistics such as min, max, mean, etc.

    Args:
        ang_err (np.array): angular error
    Returns:
        ang_err_stats (dict):  angular error statistics
    """
    ang_err = ang_err[~np.isnan(ang_err)]
    ang_err_stats = {"min": np.min(ang_err),
                     "10prc": np.percentile(ang_err, 10),
                     "median": np.median(ang_err),
                     "mean": np.mean(ang_err),
                     "90prc": np.percentile(ang_err, 90),
                     "max": np.max(ang_err)}
    return ang_err_stats

def angular_error(ground_truth_image, corrected_image, measurement_type):
    B_gt, G_gt, R_gt = cv2.split(ground_truth_image);
    B_cor, G_cor, R_cor = cv2.split(corrected_image);
    
    if measurement_type == 'mean':  
        e_gt = np.array([np.mean(B_gt), np.mean(G_gt), np.mean(R_gt)]);
        e_est = np.array([np.mean(B_cor), np.mean(G_cor), np.mean(R_cor)]);  
    
    elif measurement_type == 'median':
        e_gt = np.array([np.median(B_gt), np.median(G_gt), np.median(R_gt)]);
        e_est = np.array([np.median(B_cor), np.median(G_cor), np.median(R_cor)]);
        
    error_cos = np.dot(e_gt, e_est)/(np.linalg.norm(e_gt)*np.linalg.norm(e_est));
    e_angular = np.degrees(np.arccos(error_cos));
    return e_angular;
    
def mean_angular_error(img1, img2, mask=None):

    img1_reshaped = np.reshape(img1, (-1, 3))
    img2_reshaped = np.reshape(img2, (-1, 3))

    img1_norm = np.linalg.norm(img1_reshaped, axis=-1) + 1e-8
    img2_norm = np.linalg.norm(img2_reshaped, axis=-1) + 1e-8

    img1_unit = img1_reshaped / img1_norm[:, None]
    img2_unit = img2_reshaped / img2_norm[:, None]

    cos_similarity = (img1_unit * img2_unit).sum(axis=-1)
    cos_similarity = np.clip(cos_similarity, 0, 1)
    ang_error = np.degrees(np.arccos(cos_similarity))
    if mask is not None:
        # print(mask)
        mask_reshaped = np.reshape(mask, (-1,))
        ang_error = ang_error[mask_reshaped]
    mean_angular_error = ang_error.mean()

    return mean_angular_error
