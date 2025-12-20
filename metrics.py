import numpy as np
import sklearn as sk
from eit_image import EIT_Image
import pandas as pd

def dice_score(a,b):
    if type(a)!=np.ndarray:
        a = np.array(a)
    if type(b)!=np.ndarray:
        b = np.array(b)

    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")
    
    ni = np.sum(a*b)
    na = np.sum(a)
    nb = np.sum(b)
    
    return 2*ni/(na+nb)

def metrics_routine(pred_target_list):
    l1_err_list = []
    l2_err_list = []
    l2_rel_list = []
    l1_rel_list = []
    dice_list = []
    precision_list = []
    recall_list = []
    specificity_list = []


    for pred, target in pred_target_list:
        l1_err = np.linalg.norm(pred-target,1)
        l2_err = np.linalg.norm(pred-target)
        l2_rel_err = l2_err/np.linalg.norm(target)
        l1_rel_err = l1_err/np.linalg.norm(target,1)

        pred_bin = np.where(pred > 0.5, 1.0, 0.0)
        dscore = dice_score(pred_bin,target)
        precision = sk.metrics.precision_score(pred_bin.flatten(),target.flatten())
        recall = sk.metrics.recall_score(pred_bin.flatten(),target.flatten())
        specificity = sk.metrics.recall_score(pred_bin.flatten(),target.flatten(),pos_label=0.0)    

        l1_err_list.append(l1_err)
        l2_err_list.append(l2_err)
        l1_rel_list.append(l1_rel_err)
        l2_rel_list.append(l2_rel_err)
        dice_list.append(dscore)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)


    raw_data = {
        'l1_err': l1_err_list,
        'l2_err': l1_err_list,
        'l1_rel_err': l1_rel_list,
        'l2_rel_err:': l2_rel_list,
        'precision':precision_list,
        'recall':recall_list,
        'specificity':specificity_list,
        'dice': dice_list
    }

    results_df = pd.DataFrame(raw_data)
    return results_df
