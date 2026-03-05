import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin 
from sklearn.linear_model import RidgeClassifier, LogisticRegressionCV, LogisticRegression
from sktime.transformations.panel.rocket import Rocket
import pandas as pd
from sklearn.metrics import log_loss, confusion_matrix
#from cuml.linear_model import LogisticRegression
import cupy as cp 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix
'''
def tss_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0  # handle edge cases (e.g. single class)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr - fpr

'''
class MaskAwareROCKET(BaseEstimator, ClassifierMixin):
    """
    Mask-aware ROCKET classifier.
    X: [n_samples, n_timesteps, n_features]
    mask: same shape as X, or [n_samples, n_timesteps] (1=valid, 0=missing)
    """
    def __init__(self, num_kernels=10000, random_state=None, classifier=None, C=None):
        self.num_kernels = num_kernels
        self.random_state = random_state
#        self.classifier = classifier or RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
#        self.classifier = RidgeClassifier(alpha=1.0, class_weight='balanced')
        #self.classifier = LogisticRegression( penalty='l2', C=1.0, solver='saga', class_weight='balanced',max_iter=2000, random_state=self.random_state)
        #self.classifier = LogisticRegressionCV(max_iter=10000, penalty='l2',  class_weight='balanced') 
        #self.classifier = LogisticRegression(max_iter=10000, penalty='l2', C=0.0005, class_weight='balanced') 
        #tss_scorer = make_scorer(tss_score, greater_is_better=True)
 
        self.classifier = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=10000, 
            n_jobs=-1,
            verbose =1
        )
        self.rocket = Rocket(num_kernels=num_kernels, random_state=random_state)
        #self.optimal_threshold = 0.1  # default threshold
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
