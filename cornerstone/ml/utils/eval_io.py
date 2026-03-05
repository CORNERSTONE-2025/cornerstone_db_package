# utils/eval_io.py
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any


@torch.no_grad()
def collect_logits(model, loader, device="cuda") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Collect logits and labels from a model and data loader.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader containing batches
        device: Device to run inference on
        
    Returns:
        Tuple of (logits, labels, harp_numbers, timestamps)
        harp_numbers and timestamps are None if not available in dataset
    """
    model.eval()
    logits, ys, harps, times = [], [], [], []
    
    for batch in loader:
        # Expect (X, mask, y) in dataset; extend gracefully if extra metadata is present
        X, mask, y = batch[:3]
        X, mask = X.to(device), mask.to(device)
        z = model(X, mask)  # logits
        logits.append(z.detach().cpu().numpy().ravel())
        ys.append(y.numpy().ravel())
        
        # Optional metadata if provided by dataset
        if len(batch) > 3:
            harps.append(batch[3])
        if len(batch) > 4:
            times.append(batch[4])
    
    z = np.concatenate(logits)
    y = np.concatenate(ys)
    harp = np.concatenate(harps) if harps else None
    tstr = np.concatenate(times) if times else None
    
    return z, y, harp, tstr


def reliability_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and Brier score for reliability assessment.
    
    Uses equal-count binning for robustness. Bins are created by splitting sorted
    probabilities into n_bins groups of approximately equal size.
    
    Args:
        y: True labels (0 or 1)
        p: Predicted probabilities
        n_bins: Number of bins for ECE calculation
        
    Returns:
        Dictionary containing ECE, Brier score, bin statistics, and bin edges
    """
    # Guard against too many bins (can cause empty bins)
    n = len(p)
    n_bins = int(max(1, min(n_bins, n)))
    
    # Sort by predicted probabilities
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    
    acc, conf, frac, bin_edges = [], [], [], []
    
    for i, b in enumerate(bins):
        if len(b) == 0:
            continue
        pb, yb = p[b], y[b]
        acc.append(float(np.mean(yb)))
        conf.append(float(np.mean(pb)))
        frac.append(len(b) / len(p))
        
        # Store bin edges for reproducibility
        bin_edges.append({
            "bin": i,
            "min_prob": float(np.min(pb)),
            "max_prob": float(np.max(pb)),
            "count": len(b)
        })
    
    # Calculate ECE
    ece = float(np.sum(np.array(frac) * np.abs(np.array(acc) - np.array(conf))))
    
    # Calculate Brier score
    brier = float(np.mean((p - y) ** 2))
    
    return {
        "acc": acc,
        "conf": conf, 
        "frac": frac,
        "ECE": ece,
        "Brier": brier,
        "bin_edges": bin_edges,
        "n_bins": n_bins
    }


def save_predictions_csv(logits: np.ndarray, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        filepath: str,
                        harp_numbers: Optional[np.ndarray] = None,
                        timestamps: Optional[np.ndarray] = None,
                        prob_cal: Optional[np.ndarray] = None,
                        prob_raw: Optional[np.ndarray] = None) -> None:
    """
    Save predictions to CSV file with optional metadata.
    
    Args:
        logits: Raw model logits
        y_true: True labels
        y_pred: Predicted binary labels
        filepath: Output file path
        harp_numbers: Optional HARP numbers
        timestamps: Optional timestamps
        prob_cal: Optional calibrated probabilities
        prob_raw: Optional raw sigmoid probabilities
    """
    # Create data array
    if harp_numbers is not None:
        data = np.column_stack([harp_numbers, y_true, logits])
        header = "harp_num,y_true,logits"
    else:
        data = np.column_stack([np.arange(len(y_true)), y_true, logits])
        header = "sample_id,y_true,logits"
    
    if prob_cal is not None:
        data = np.column_stack([data, prob_cal])
        header += ",prob_cal"
    
    if prob_raw is not None:
        data = np.column_stack([data, prob_raw])
        header += ",prob_raw"
    
    if y_pred is not None:
        data = np.column_stack([data, y_pred])
        header += ",y_pred"
    
    if timestamps is not None:
        data = np.column_stack([data, timestamps])
        header += ",timestamp"
    
    # Convert to string to handle mixed dtypes cleanly
    data_str = data.astype(str)
    
    # Save to CSV
    np.savetxt(filepath, data_str, delimiter=",", header=header, comments="", fmt="%s")


def expected_confusion(p_test: np.ndarray, tau_star: float) -> tuple:
    """
    Compute expected confusion matrix from calibrated probabilities and threshold.
    
    Args:
        p_test: Calibrated test probabilities
        tau_star: Optimal threshold
        
    Returns:
        Tuple of (TP, FP, FN, TN) counts
    """
    y_pred = (p_test >= tau_star).astype(int)
    
    # For expected counts, we can use the probabilities directly
    # Expected TP = sum of probabilities for positive predictions
    # Expected FP = sum of (1 - probabilities) for positive predictions
    # Expected FN = sum of probabilities for negative predictions  
    # Expected TN = sum of (1 - probabilities) for negative predictions
    
    pos_mask = y_pred == 1
    neg_mask = y_pred == 0
    
    TP = float(np.sum(p_test[pos_mask]))
    FP = float(np.sum(1 - p_test[pos_mask]))
    FN = float(np.sum(p_test[neg_mask]))
    TN = float(np.sum(1 - p_test[neg_mask]))
    
    return TP, FP, FN, TN

