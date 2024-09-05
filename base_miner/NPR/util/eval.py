import numpy as np


def compute_tfpn(y_true, y_pred):
    
    tp = sum(y_pred[y_true==1] > 0.5)
    fp = sum(y_pred[y_true==0] > 0.5)
    tn = sum(y_pred[y_true==0] <= 0.5)
    fn = sum(y_pred[y_true==1] <= 0.5)
    return tp, fp, tn, fn


def compute_metrics(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score