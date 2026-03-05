import os
import h5py
import numpy as np
import torch
from utils.datasets import FlareHDF5Dataset

def get_splits(data_path, n_folds=5, label_type='label_Mplus_delta24', use_mask=True, lazy_load=False):
    """
    Load existing dataset splits from HDF5 files or create them if they don't exist.
    The splits are created based on HARPs (Active Region numbers) and use the maximum sequence length (121 timesteps).
    The actual window size is handled by the dataset's __getitem__ method.
    
    Args:
        data_path: Path to the main HDF5 file
        n_folds: Number of cross-validation folds
        label_type: Type of label to use (e.g., 'label_Mplus_delta24')
        use_mask: Whether to use masks for missing values
        lazy_load: Whether to load data lazily
        
    Returns:
        Dictionary containing train, validation, and test datasets
    """
    splits_dir = os.path.join(os.path.dirname(data_path), 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # Check if all required split files exist
    test_path = os.path.join(splits_dir, 'M+_flare_test.h5')
    required_files = [test_path]
    for fold in range(n_folds):
        required_files.extend([
            os.path.join(splits_dir, f'M+_flare_cv_{fold}_train.h5'),
            os.path.join(splits_dir, f'M+_flare_cv_{fold}_val.h5')
        ])
    
    # Check if all required files exist
    all_files_exist = all(os.path.exists(f) for f in required_files)
    
    if not all_files_exist:
        print("Creating new splits...")
        # Create splits using the dataset class with maximum sequence length
        splits = FlareHDF5Dataset.create_cv_datasets(
            h5_path=data_path,
            n_folds=n_folds,
            random_state=42,
            label_type=label_type,
            test_size=0.15,
            val_size=0.2,
            test_split_strategy='chronological',
            use_mask=use_mask,
            lazy_load=lazy_load,
            timesteps=121  # Maximum sequence length
        )
        
        # Save test dataset
        print(f"Saving test dataset to {test_path}")
        with h5py.File(test_path, 'w') as f:
            for key, value in splits['test'].__dict__.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    f.create_dataset(key, data=value.numpy() if isinstance(value, torch.Tensor) else value)
        
        # Save train and validation datasets for each fold
        for fold in range(n_folds):
            train_path = os.path.join(splits_dir, f'M+_flare_cv_{fold}_train.h5')
            val_path = os.path.join(splits_dir, f'M+_flare_cv_{fold}_val.h5')
            
            print(f"Saving fold {fold} train dataset to {train_path}")
            with h5py.File(train_path, 'w') as f:
                for key, value in splits['train'][fold].__dict__.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        f.create_dataset(key, data=value.numpy() if isinstance(value, torch.Tensor) else value)
            
            print(f"Saving fold {fold} validation dataset to {val_path}")
            with h5py.File(val_path, 'w') as f:
                for key, value in splits['val'][fold].__dict__.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        f.create_dataset(key, data=value.numpy() if isinstance(value, torch.Tensor) else value)
    else:
        print("Loading existing splits...")
    
    # Load test dataset with maximum sequence length
    test_dataset = FlareHDF5Dataset(test_path, label_type=label_type, use_mask=use_mask, lazy_load=lazy_load, timesteps=121)
    
    # Load train and validation datasets for each fold with maximum sequence length
    train_datasets = []
    val_datasets = []
    
    for fold in range(n_folds):
        train_path = os.path.join(splits_dir, f'M+_flare_cv_{fold}_train.h5')
        val_path = os.path.join(splits_dir, f'M+_flare_cv_{fold}_val.h5')
        
        train_dataset = FlareHDF5Dataset(train_path, label_type=label_type, use_mask=use_mask, lazy_load=lazy_load, timesteps=121)
        val_dataset = FlareHDF5Dataset(val_path, label_type=label_type, use_mask=use_mask, lazy_load=lazy_load, timesteps=121)
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    
    return {
        'train': train_datasets,
        'val': val_datasets,
        'test': test_dataset
    } 