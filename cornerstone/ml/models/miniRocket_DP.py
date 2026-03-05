import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sktime.transformations.panel.rocket import MiniRocket
#from cuml.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cupy as cp
from sklearn.metrics import make_scorer

 

class MaskAwareMINIROCKET(BaseEstimator, ClassifierMixin):
    def __init__(self, num_kernels=2000, random_state=None, classifier=None, C=None):
        self.num_kernels = num_kernels
        self.random_state = random_state

        self.classifier = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=10000, 
            n_jobs=-1,
            verbose =1
        )
        
        self.rocket = MiniRocket(num_kernels=num_kernels, random_state=random_state)
        self.scaler = StandardScaler()
 
 
 
    def _to_sktime_panel(self, X):
        return [pd.DataFrame(X[i, :, :]) for i in range(X.shape[0])]

    def fit(self, X, y, mask=None):  
            
        cp._default_memory_pool.free_all_blocks()  # release GPU memory
        X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        X = np.clip(X, -1e3, 1e3)

        X_panel = self._to_sktime_panel(X)
        self.rocket.fit(X_panel, y)
        X_transformed = self.rocket.transform(X_panel)

        X_scaled = self.scaler.fit_transform(X_transformed)
        '''
        X_gpu = cp.asarray(X_scaled)
        y_gpu = cp.asarray(y.astype(np.int32))

        self.classifier.fit(X_gpu, y_gpu)
        '''
        self.classifier.fit(X_scaled, y)
        
        # Free GPU memory used temporarily
        '''del X_gpu, y_gpu
        cp._default_memory_pool.free_all_blocks()
        '''
        return self

    def _transform(self, X ):
        X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        X = np.clip(X, -1e3, 1e3)

        X_panel = self._to_sktime_panel(X)
        X_transformed = self.rocket.transform(X_panel)
        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled

    def predict(self, X, mask=None):
        x_trans = self._transform(X)
        y_pred = self.classifier.predict(x_trans)#cp.asnumpy()
        '''
        X_gpu = self._transform(X)
        y_pred = cp.asnumpy(self.classifier.predict(X_gpu))
        del X_gpu
        cp._default_memory_pool.free_all_blocks()
        '''
        return y_pred

    def predict_proba(self, X, mask=None):
        #X_gpu = self._transform(X)
        x_trans = self._transform(X)
        proba = self.classifier.predict_proba(x_trans)#cp.asnumpy()
        '''del X_gpu
        cp._default_memory_pool.free_all_blocks()
        '''
        return proba