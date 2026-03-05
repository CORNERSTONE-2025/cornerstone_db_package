# utils/metrics.py
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef

def compute_mcc(y_true, y_pred):
    """
    Compute Matthews Correlation Coefficient (MCC).
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        MCC value between -1 and 1
    """
    return matthews_corrcoef(y_true, y_pred)

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred_probs = np.nan_to_num(y_pred_probs, nan=0.0)
    y_pred = (y_pred_probs >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    mcc = compute_mcc(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = tpr - fpr
    denom = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = (2 * (tp * tn - fp * fn)) / denom if denom > 0 else 0
    return {"F1": f1, "AUC": auc, "MCC": mcc, "TSS": tss, "HSS": hss, "TP": tp, "TN": tn, "FP": fp, "FN": fn}

def compute_pos_weight(dataset):
    labels = [int(y.item()) for _, _, y in dataset]
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0:
        print("⚠️ No positive samples in this dataset — using neutral pos_weight = 1.0")
        return torch.tensor([1.0])
    return torch.tensor([neg / pos])

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj
