# utils/threshold_optimizer.py
import numpy as np
import torch
from typing import Dict, List, Tuple, Callable, Optional
from .metrics import compute_metrics, compute_mcc

class ThresholdOptimizer:
    """
    Modular threshold optimizer for binary classification.
    
    Supports different optimization metrics and can be easily extended
    with custom metric functions.
    """
    
    def __init__(self, n_thresholds: int = 100, threshold_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize threshold optimizer.
        
        Args:
            n_thresholds: Number of threshold values to test
            threshold_range: Range of thresholds to test (min, max)
        """
        self.n_thresholds = n_thresholds
        self.threshold_range = threshold_range
        self.thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        # Predefined metric functions
        self.metric_functions = {
            'TSS': self._compute_tss,
            'F1': self._compute_f1,
            'MCC': self._compute_mcc,
            'precision': self._compute_precision,
            'recall': self._compute_recall,
            'precision_at_recall': self._compute_precision_at_recall,
            'f1_at_precision': self._compute_f1_at_precision
        }
    
    def optimize_threshold(self, 
                          y_true: np.ndarray, 
                          y_probs: np.ndarray, 
                          metric: str = 'TSS',
                          **metric_kwargs) -> Dict:
        """
        Find optimal threshold for given metric.
        
        Args:
            y_true: True labels (0 or 1)
            y_probs: Predicted probabilities
            metric: Metric to optimize ('TSS', 'F1', 'MCC', 'precision', 'recall', etc.)
            **metric_kwargs: Additional arguments for metric function
            
        Returns:
            Dictionary containing optimal threshold and corresponding metrics
        """
        if metric not in self.metric_functions:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(self.metric_functions.keys())}")
        
        metric_fn = self.metric_functions[metric]
        best_threshold = None
        best_metric_value = -np.inf if metric in ['TSS', 'F1', 'MCC', 'precision', 'recall'] else np.inf
        best_metrics = None
        all_results = []
        
        for threshold in self.thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            metrics = compute_metrics(y_true, y_pred)
            
            # Compute the optimization metric
            metric_value = metric_fn(y_true, y_pred, **metric_kwargs)
            
            all_results.append({
                'threshold': threshold,
                'metric_value': metric_value,
                'metrics': metrics
            })
            
            # Update best if better
            if metric in ['TSS', 'F1', 'MCC', 'precision', 'recall']:
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = threshold
                    best_metrics = metrics.copy()
            else:
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = threshold
                    best_metrics = metrics.copy()
        
        return {
            'optimal_threshold': best_threshold,
            'optimal_metric_value': best_metric_value,
            'optimal_metrics': best_metrics,
            'all_results': all_results,
            'metric_used': metric
        }
    
    def _compute_tss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute True Skill Statistic (TSS)."""
        tn, fp, fn, tp = self._get_confusion_matrix(y_true, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr - fpr
    
    def _compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute F1 score."""
        tn, fp, fn, tp = self._get_confusion_matrix(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    def _compute_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient (MCC)."""
        return compute_mcc(y_true, y_pred)
    
    def _compute_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute precision."""
        tn, fp, fn, tp = self._get_confusion_matrix(y_true, y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def _compute_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute recall."""
        tn, fp, fn, tp = self._get_confusion_matrix(y_true, y_pred)
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def _compute_precision_at_recall(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   min_recall: float = 0.8) -> float:
        """Compute precision at minimum recall threshold."""
        recall = self._compute_recall(y_true, y_pred)
        if recall >= min_recall:
            return self._compute_precision(y_true, y_pred)
        else:
            return 0.0  # Penalize if recall is too low
    
    def _compute_f1_at_precision(self, y_true: np.ndarray, y_pred: np.ndarray,
                               min_precision: float = 0.8) -> float:
        """Compute F1 score at minimum precision threshold."""
        precision = self._compute_precision(y_true, y_pred)
        if precision >= min_precision:
            return self._compute_f1(y_true, y_pred)
        else:
            return 0.0  # Penalize if precision is too low
    
    def _get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        """Get confusion matrix components."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return tn, fp, fn, tp
    
    def add_custom_metric(self, name: str, metric_fn: Callable):
        """
        Add a custom metric function.
        
        Args:
            name: Name of the metric
            metric_fn: Function that takes (y_true, y_pred, **kwargs) and returns a float
        """
        self.metric_functions[name] = metric_fn
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        return list(self.metric_functions.keys())


def create_threshold_optimizer(n_thresholds: int = 100, 
                             threshold_range: Tuple[float, float] = (0.0, 1.0)) -> ThresholdOptimizer:
    """
    Factory function to create a threshold optimizer with default settings.
    
    Args:
        n_thresholds: Number of threshold values to test
        threshold_range: Range of thresholds to test (min, max)
        
    Returns:
        ThresholdOptimizer instance
    """
    return ThresholdOptimizer(n_thresholds=n_thresholds, threshold_range=threshold_range) 