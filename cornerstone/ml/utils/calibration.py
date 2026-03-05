# utils/calibration.py
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize


class PlattCalibrator:
    """
    Platt scaling calibrator for probability calibration.
    
    Fits a logistic regression model to map raw logits to calibrated probabilities.
    This method preserves ranking while improving calibration.
    """
    
    def __init__(self, class_weight="balanced", C=1.0, max_iter=1000):
        """
        Initialize Platt calibrator.
        
        Args:
            class_weight: Class weight strategy for logistic regression
            C: Regularization strength (inverse of regularization)
            max_iter: Maximum number of iterations for optimization
        """
        self.lr = LogisticRegression(
            solver="lbfgs", 
            class_weight=class_weight,
            C=C, 
            max_iter=max_iter
        )
        self.fitted = False
    
    def fit(self, z_val: np.ndarray, y_val: np.ndarray):
        """
        Fit calibrator on validation logits and labels.
        
        Args:
            z_val: Validation logits (1D array)
            y_val: Validation labels (1D array, 0 or 1)
            
        Returns:
            self: Fitted calibrator
        """
        # Check if we need to flip logits for monotonic increasing mapping
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val, z_val)
            self.flip = (auc < 0.5)
        except Exception:
            self.flip = False
        
        z = (-z_val if self.flip else z_val).reshape(-1, 1)
        self.lr.fit(z, y_val.astype(int))
        self.fitted = True
        return self
    
    def transform(self, z: np.ndarray) -> np.ndarray:
        """
        Transform logits to calibrated probabilities.
        
        Args:
            z: Logits to calibrate (1D array)
            
        Returns:
            Calibrated probabilities (1D array)
        """
        assert self.fitted, "Calibrator must be fitted before transform"
        if getattr(self, "flip", False):
            # Model was fit on -z; keep monotone-increasing mapping vs original z
            p = self.lr.predict_proba((-z).reshape(-1, 1))[:, 1]
            return 1.0 - p
        else:
            return self.lr.predict_proba(z.reshape(-1, 1))[:, 1]
    
    def state_dict(self):
        """Get calibrator state for serialization."""
        return {
            "type": "platt",
            "coef": float(self.lr.coef_.ravel()[0]),
            "intercept": float(self.lr.intercept_.ravel()[0]),
            "flip": bool(getattr(self, "flip", False))
        }


class TemperatureCalibrator:
    """
    Temperature scaling calibrator for probability calibration.
    
    Fits a single temperature parameter T to scale logits: p = sigmoid(z/T).
    This method preserves ranking while improving calibration.
    """
    
    def __init__(self):
        """Initialize temperature calibrator."""
        self.T = 1.0
        self.fitted = False
    
    def fit(self, z_val: np.ndarray, y_val: np.ndarray):
        """
        Fit calibrator on validation logits and labels.
        
        Args:
            z_val: Validation logits (1D array)
            y_val: Validation labels (1D array, 0 or 1)
            
        Returns:
            self: Fitted calibrator
        """
        def nll(logT):
            """Negative log-likelihood for temperature optimization."""
            T = np.exp(logT)
            p = sigmoid(z_val / T)
            eps = 1e-12
            return -np.mean(
                y_val * np.log(p + eps) + (1 - y_val) * np.log(1 - p + eps)
            )
        
        res = minimize(nll, x0=0.0, method="L-BFGS-B")
        if not res.success:
            # Fallback: T=1.0 (no change) if optimization fails
            print(f"⚠️ Temperature optimization failed (status: {res.status}, message: {res.message}), using T=1.0 (no calibration)")
            self.T = 1.0
            self.temp_opt_status = res.status
        else:
            self.T = float(np.exp(res.x[0]))
            self.temp_opt_status = "success"
        self.fitted = True
        return self
    
    def transform(self, z: np.ndarray) -> np.ndarray:
        """
        Transform logits to calibrated probabilities.
        
        Args:
            z: Logits to calibrate (1D array)
            
        Returns:
            Calibrated probabilities (1D array)
        """
        assert self.fitted, "Calibrator must be fitted before transform"
        return sigmoid(z / self.T)
    
    def state_dict(self):
        """Get calibrator state for serialization."""
        return {
            "type": "temperature", 
            "T": float(self.T),
            "temp_opt_status": getattr(self, "temp_opt_status", "unknown")
        }


def create_calibrator(kind: str = "platt"):
    """
    Factory function to create calibrator instances.
    
    Args:
        kind: Type of calibrator ("platt", "logistic", "temp", "temperature", "auto")
        
    Returns:
        Calibrator instance
        
    Raises:
        ValueError: If unknown calibrator kind is specified
    """
    if kind.lower() in ["platt", "logistic"]:
        return PlattCalibrator()
    elif kind.lower() in ["temp", "temperature"]:
        return TemperatureCalibrator()
    elif kind.lower() == "auto":
        # Auto-choose based on positive samples (will be set later)
        return None  # Will be determined at fit time
    else:
        raise ValueError(f"Unknown calibrator kind: {kind}. Available: platt, temperature, auto")


def prior_adjust(p_cal, pi_val, pi_test, eps=1e-9):
    """
    Adjust calibrated probabilities for prior shift between validation and test sets.
    
    This is useful for analysis to separate calibration failure from pure base-rate change.
    Uses logit-space arithmetic for numerical stability at extremes.
    
    Args:
        p_cal: Calibrated probabilities
        pi_val: Positive class proportion in validation set
        pi_test: Positive class proportion in test set
        eps: Small value to avoid log(0)
        
    Returns:
        Prior-adjusted probabilities
    """
    p = np.clip(p_cal, eps, 1 - eps)
    
    def logit(x):
        return np.log(x) - np.log1p(-x)
    
    delta = logit(np.clip(pi_test, eps, 1 - eps)) - logit(np.clip(pi_val, eps, 1 - eps))
    return sigmoid(logit(p) + delta)


def load_calibrator(state):
    """
    Reconstruct calibrator from serialized state.
    
    Args:
        state: Dictionary containing calibrator state (from state_dict())
        
    Returns:
        Reconstructed calibrator instance or None if unknown type
    """
    t = state.get("type")
    
    if t == "platt":
        c = PlattCalibrator()
        c.fitted = True
        c.flip = state.get("flip", False)
        
        # Rebuild sklearn LogisticRegression
        from sklearn.linear_model import LogisticRegression
        c.lr = LogisticRegression()
        c.lr.classes_ = np.array([0, 1])
        c.lr.coef_ = np.array([[float(state["coef"])]])
        c.lr.intercept_ = np.array([float(state["intercept"])])
        c.lr.n_features_in_ = 1
        c.lr.n_iter_ = np.array([1], dtype=int)
        
        return c
    
    elif t == "temperature":
        c = TemperatureCalibrator()
        c.T = float(state["T"])
        c.fitted = True
        c.temp_opt_status = state.get("temp_opt_status", "unknown")
        
        return c
    
    return None

