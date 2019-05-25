import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score



def get_iou(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return jaccard_score(y_true, y_pred)

def get_precision(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return precision_score(y_true, y_pred)

def get_recall(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return recall_score(y_true, y_pred)
