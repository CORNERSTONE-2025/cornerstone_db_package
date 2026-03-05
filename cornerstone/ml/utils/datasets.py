# utils/datasets.py
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

class FlareHDF5Dataset(Dataset):
    def __init__(self, h5_path, timesteps=121):
        with h5py.File(h5_path, "r") as hf:
            self.X = hf["X"][:][:, -timesteps:, :] 
            self.mask = ~np.isnan(self.X)
            self.Y = hf["Y"][:].astype(str)
            valid = (self.mask.sum(axis=(1, 2)) > 0)
            print(f"⚠️ Dropped {(~valid).sum()} empty samples out of {len(valid)}")
            self.X = self.X[valid]
            self.mask = self.mask[valid]
            self.Y = self.Y[valid]
        self.X = torch.tensor(self.X).float()
        self.mask = torch.tensor(self.mask).float()
        print(h5_path)
        print(np.unique(self.Y))
        self.Y = torch.tensor((self.Y == "YES").astype(float))
        print(np.unique(self.Y)) 
         
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.Y[idx]
    
    def get_numpy_with_mask(self): #DP
        """
        Return the dataset as NumPy arrays: X, y, mask
        - X: [n_samples, T, d] (features)
        - y: [n_samples] (labels)
        - mask: [n_samples, T, d] or [n_samples, T] indicating valid timesteps
        """
        X_list = []
        y_list = []
        mask_list = []

        for i in range(len(self)):
            x, y, mask = self[i]  # same as __getitem__
            X_list.append(x.numpy() if isinstance(x, torch.Tensor) else x)
            y_list.append(y.numpy() if isinstance(y, torch.Tensor) else y)
            mask_list.append(mask.numpy() if isinstance(mask, torch.Tensor) else mask)

        X = np.stack(X_list)
        y = np.array(y_list)
        mask = np.stack(mask_list)
        return X, y, mask
