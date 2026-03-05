import json
import numpy as np
from typing import Dict, List, Any, Optional
from .metrics import compute_metrics, convert_to_serializable
from .path_manager import PathManager

class MetricsManager:
    """Manages metrics computation and storage."""
    
    def __init__(self, path_manager: Optional[PathManager] = None):
        self.path_manager = path_manager or PathManager()
        
    def compute_metrics_with_threshold(self, y_true: np.ndarray, predictions: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Computes metrics for predictions with optional thresholding."""
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        # Check if predictions are already binary (only 0s and 1s)
        if np.all(np.isin(predictions, [0, 1])):
            # Predictions are already binary (0 or 1)
            return compute_metrics(y_true, predictions)
        else:
            # Predictions are probabilities, apply threshold
            y_pred = (predictions >= threshold).astype(int)
            return compute_metrics(y_true, y_pred)
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str) -> None:
        """Saves metrics to a JSON file."""
        metrics_path = self.path_manager.get_metrics_path(filename)
        with open(metrics_path, "w") as f:
            json.dump(convert_to_serializable(metrics), f, indent=2)
    
    def load_metrics(self, filename: str) -> Dict[str, Any]:
        """Loads metrics from a JSON file."""
        metrics_path = self.path_manager.get_metrics_path(filename)
        with open(metrics_path, "r") as f:
            return json.load(f)
    
    def get_metrics_summary(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Computes summary statistics for metrics across windows."""
        summary = {}
        
        # Get all metric names from the test metrics of the first window
        first_window = list(metrics.keys())[0]
        if "test" not in metrics[first_window]:
            print(f"⚠️ No test metrics found for window {first_window}")
            return {}
        
        metric_names = metrics[first_window]["test"].keys()
        
        for metric in metric_names:
            values = []
            for window_metrics in metrics.values():
                if "test" in window_metrics and metric in window_metrics["test"]:
                    values.append(window_metrics["test"][metric])
            
            if values:  # Only calculate stats if we have values
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return summary 