# xgboost_wrapper.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class MaskAwareXGBoost(BaseEstimator, ClassifierMixin):
    """
    Wrapper for XGBoost binary classification in a similar style as MaskAwareROCKET.
    
    X: [n_samples, n_timesteps, n_features]
    mask: optional mask of same shape as X or [n_samples, n_timesteps] (1=valid, 0=missing)
    """
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=None, mask_concat=True, weight_scale=None, reg_lambda =None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.mask_concat = mask_concat
        
        '''if reg_lambda is not None:
            reg_lambda =1/reg_lambda
        '''
        self.scaler = StandardScaler()
        self.clf = xgb.XGBClassifier(
            scale_pos_weight = weight_scale,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            reg_lambda=reg_lambda,      # L2 regularization (default 1.0)
            reg_alpha=0,       # L1 regularization (default 0.0)
            learning_rate=self.learning_rate,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state
        )

    def _flatten(self, X):
        """Flatten 3D array to 2D and apply mask if provided.""" 
        X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        X = np.clip(X, -1e3, 1e3)
        # Flatten time dimension
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, n_timesteps * n_features )
        return X_flat

    def fit(self, X, y, mask=None):
        if mask is not None and self.mask_concat: 
            X = np.concatenate([X, mask], axis=-1) 
        X_flat = self._flatten(X)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.clf.fit(X_scaled, y)
        return self

    def predict(self, X, mask=None):
        if mask is not None and self.mask_concat: 
            X = np.concatenate([X, mask], axis=-1) 
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X, mask=None):
        if mask is not None and self.mask_concat: 
            X = np.concatenate([X, mask], axis=-1) 
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.clf.predict_proba(X_scaled)
