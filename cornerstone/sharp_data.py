import os
import numpy as np
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, Subset
from datetime import datetime
from typing import Optional, List, Dict, Union, Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from collections import Counter

class SharpDataset(Dataset):
    """Base dataset class for loading HDF5 data for flare forecasting.
    
    This class handles the core functionality of loading and accessing HDF5 data.
    It provides a clean interface for accessing features, labels, and metadata.
    Compatible with the original FlareHDF5Dataset's interface for training pipeline compatibility.
    
    Attributes:
        h5_path (str): Path to the HDF5 file
        timesteps (int): Number of timesteps to use
        label: Base label name (e.g., 'Mplus_delta24')
        use_mask (bool): Whether to use mask for missing values
        multi_class (bool): Whether to use multi-class labels
        feature_names (List[str]): List of feature names
        harp_numbers (np.ndarray): Array of HARP numbers
        timestamps (np.ndarray): Array of timestamps
        X (torch.Tensor): Feature tensor of shape (n_samples, n_timesteps, n_features)
        mask (torch.Tensor): Mask tensor for missing values
        Y (torch.Tensor): Label tensor (binary or multi-class)
        label_map (Dict[str, int]): Mapping from label strings to indices when multi_class=True
    """
    
    def __init__(
        self,
        h5_path: str,
        timesteps: int = 121,
        label: str = None,  # e.g., 'Mplus_delta24'
        use_mask: bool = True,
        multi_class: bool = False
    ):
        """Initialize the dataset.
        
        Args:
            h5_path: Path to the HDF5 file
            timesteps: Number of timesteps to use
            label: Base label name (e.g., 'Mplus_delta24')
            use_mask: Whether to use mask for missing values
            multi_class: If True, use decomposed labels (C,M,X,NO); if False, use binary labels (YES,NO)
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {h5_path}\n"
                f"Make sure you've run dataset_compilation/extract_sharp_timeseries.py first"
            )
        self.h5_path = h5_path
        self.timesteps = timesteps
        self.label = label
        self.use_mask = use_mask
        self.multi_class = multi_class
        
        
        # Load all data at initialization
        with h5py.File(h5_path, 'r') as f:
            # Load feature names
            raw_names = f.attrs["feature_names"]
            
            if isinstance(raw_names, bytes):
                raw_names = raw_names.decode("utf-8") 
            self.feature_names = raw_names.split(',')
            
            # Load data - ensure format is (samples, timesteps, features)
            raw_X = f['X'][:] 
            if raw_X.shape[1] == len(self.feature_names):
                # Data is (samples, features, timesteps) - need to transpose
                raw_X = raw_X.transpose(0, 2, 1)  # Now (samples, timesteps, features)
            
            # Load timestamps and labels
            self.harp_numbers = f['HARPNUM'][:]
            self._raw_timestamps = f['timestamps'][:]
            self._raw_issuing_times = [ts[-1] for ts in self.timestamps]
            if label is None:
                raise ValueError("You must specify the label base, e.g., label='Mplus_delta24'")
            compact_name = f'label_{label}'
            #decomposed_name = f'label_u_{label}'
            for k in f.keys() :
                print(k) 
            available_labels = [k for k in f.keys() if k.startswith('label')]
            print(compact_name)
            #print(decomposed_name)
            print(f)
            if compact_name not in f: # or decomposed_name not in f
                raise ValueError(f"label '{compact_name}'  not found.")#or '{decomposed_name}' Available: {available_labels}
            self._raw_labels = np.array([self.decode_if_bytes(y) for y in f[compact_name][:]])
            #self._raw_labels = np.array([self.decode_if_bytes(y) for y in f[decomposed_name][:]])
            
            # Check if we have enough timesteps
            available_timesteps = raw_X.shape[1]
            if timesteps > available_timesteps:
                print(f"Warning: Requested {timesteps} timesteps but only {available_timesteps} available. Using all available timesteps.")
                self.timesteps = available_timesteps
            else:
                self.timesteps = timesteps
            
            # Take only the last N timesteps
            if self.timesteps < available_timesteps:
                start_idx = available_timesteps - self.timesteps
                self.X = torch.from_numpy(raw_X[:, start_idx:, :]).float()
                self._raw_timestamps = self._raw_timestamps[:, start_idx:]
            else:
                self.X = torch.from_numpy(raw_X).float()
            
            # Create mask and convert labels
            if self.use_mask:
                self.mask = ~torch.isnan(self.X)
            else:
                self.mask = torch.ones_like(self.X, dtype=torch.bool)
            
            # Handle labels based on multi_class flag
            if self.multi_class:
                # Use decomposed labels (C, M, X, NO) - compress all NO types into single NO
                compressed_labels = []
                for lbl in self._raw_labels:
                    if lbl.startswith('NO'):
                        compressed_labels.append('NO')
                    else:
                        compressed_labels.append(lbl)
                
                unique_labels = np.unique(compressed_labels)
                # Ensure 'NO' is always index 0
                self.label_map = {}
                if 'NO' in unique_labels:
                    self.label_map['NO'] = 0
                    other_labels = [lbl for lbl in unique_labels if lbl != 'NO']
                    for i, lbl in enumerate(other_labels):
                        self.label_map[lbl] = i + 1
                else:
                    # Fallback if no 'NO' label
                    self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                self.Y = torch.tensor([self.label_map[lbl] for lbl in compressed_labels], dtype=torch.long)
            else:
                # Use binary labels (YES, NO1, NO2, NO3, NO4)
                # Convert all NO types to a single 'NO' class for binary classification
                binary_labels = []
                for lbl in self._raw_labels:
                    if lbl == 'YES':
                        binary_labels.append('YES')
                    else:
                        # All NO1, NO2, NO3, NO4 become 'NO'
                        binary_labels.append('NO')
                self.Y = torch.tensor((np.array(binary_labels) == 'YES').astype(float)).unsqueeze(1)
            
            # Filter out completely empty samples
            valid = (self.mask.sum(dim=(1, 2)) > 0)
            if not valid.all():
                self.X = self.X[valid]
                self.mask = self.mask[valid]
                self.Y = self.Y[valid]
                self.harp_numbers = self.harp_numbers[valid]
                self._raw_timestamps = self._raw_timestamps[valid]
                self._raw_labels = self._raw_labels[valid]
                self._raw_labels = self._raw_labels[valid]
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Get a sample from the dataset.
        
        Returns a tuple of (X, mask, Y) if use_mask=True, or (X, Y) if use_mask=False.
        The Y tensor is already in the correct shape for binary cross-entropy loss (n_samples, 1).
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            If use_mask=True:
                Tuple containing (X, mask, Y)
            If use_mask=False:
                Tuple containing (X, Y)
        """
        if self.use_mask:
            return self.X[idx], self.mask[idx], self.Y[idx]
        return self.X[idx], self.Y[idx]
    
    @staticmethod
    def decode_if_bytes(val: Union[bytes, str]) -> str:
        """Decode bytes to string if necessary."""
        if isinstance(val, (bytes, np.bytes_)):
            return str(val, encoding="utf-8")
        return str(val)
    
    @property
    def timestamps(self) -> np.ndarray:
        """Get timestamps as datetime objects."""
        return np.array([
            [datetime.fromisoformat(self.decode_if_bytes(t).replace(" ", "T")) 
             for t in ts_sample]
            for ts_sample in self._raw_timestamps
        ], dtype=object)
    
    def overview(self) -> None:
        """Print dataset statistics and validation information."""
        print(f"Total samples: {len(self)}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of timesteps: {self.timesteps}")
        print(f"Number of unique HARPs: {len(np.unique(self.harp_numbers))}")


        issuing_times = [ts[-1] for ts in self.timestamps]
        issuing_times = pd.to_datetime(issuing_times)
        print("\nTemporal Coverage:")
        print(f"  - Start: {issuing_times.min()}")
        print(f"  - End: {issuing_times.max()}")
        print(f"  - Duration: {issuing_times.max() - issuing_times.min()}")
        print(f"  - Total samples: {len(issuing_times)}")


        
        # Print label distribution
        unique_labels, counts = np.unique(self._raw_labels, return_counts=True)
        print("\nLabel distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(self)) * 100
            print(f"  {label}: {count} samples ({percentage:.2f}%)")
            
        '''    
        # Print decomposed label distribution
        unique_labels_u, counts_u = np.unique(self._raw_labels, return_counts=True)
        print("\nDecomposed label distribution:")
        for label, count in zip(unique_labels_u, counts_u):
            percentage = (count / len(self)) * 100
            print(f"  {label}: {count} samples ({percentage:.2f}%)")
            '''
            
        if self.multi_class:
            print("\nLabel mapping:")
            # Sort by index for consistent display
            sorted_mapping = sorted(self.label_map.items(), key=lambda x: x[1])
            for label, idx in sorted_mapping:
                print(f"  {label} -> {idx}")
        
        # Add features check
        if len(self.feature_names) == 0:
            print("Warning: No features found in dataset")
        
        # Check for reasonable time ranges
        if hasattr(self, 'timestamps'):
            start_date = min(ts[0] for ts in self.timestamps)
            end_date = max(ts[-1] for ts in self.timestamps)
            print(f"Time range: {start_date} to {end_date}")
    


class Transform(ABC):
    """Base class for all transforms."""
    
    def __init__(self):
        self.dataset = None
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply the transform to the dataset."""
        self.dataset = dataset
        return self
    
    @abstractmethod
    def transform_sample(self, sample: dict) -> dict:
        """Transform a single sample."""
        pass

class FeatureSelector(Transform):  #TODO Update the feature groups according to the final version of the dataset (to be determined yet)
    """Transform to select specific features from the dataset.
    
    Args:
        selected_features: List of feature names to keep, OR
        criterion: Predefined feature group ('physical_features', 'magnetic_features', etc.)
    """
    
    # Predefined feature groups for solar flare forecasting
    FEATURE_GROUPS = {
        # New physically meaningful feature groups
        'magnetic_flux_energy': [
            'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'TOTPOT', 'MEANPOT'
        ],
        'electric_current': [
            'MEANJZD', 'MEANJZH', 'ABSNJZH'
        ],
        'spatial_complexity': [
            'LATDTMIN', 'LATDTMAX', 'LONDTMIN', 'LONDTMAX', 'LON_FWT'
        ],
        'magnetic_field_strength': [
            'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH'
        ],
        'magnetic_shear_twist': [
            'MEANALP', 'MEANSHR', 'SHRGT45'
        ],
        'doppler_observations': [
            'OBS_VR'
        ],
        'magnetic_neutral_line': [
            'SAVNCPP', 'R_VALUE'
        ],
        'comprehensive_physical': [
            # Magnetic Flux and Energy Measures
            'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'TOTPOT', 'MEANPOT',
            # Electric Current Indicators
            'MEANJZD', 'MEANJZH', 'ABSNJZH',
            # Spatial Complexity and Location
            'LATDTMIN', 'LATDTMAX', 'LONDTMIN', 'LONDTMAX', 'LON_FWT',
            # Magnetic Field Strength and Components
            'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH',
            # Magnetic Shear and Twist
            'MEANALP', 'MEANSHR', 'SHRGT45',
            # Doppler Observations
            'OBS_VR',
            # Magnetic Neutral Line Measures
            'SAVNCPP', 'R_VALUE'
        ]
    }
    
    @classmethod
    def get_feature_group(cls, criterion: str) -> List[str]:
        """
        Get feature group based on criterion.
        
        Args:
            criterion: Feature selection criterion ('physical_features', 'magnetic_features', etc.)
            
        Returns:
            List of selected features
        """
        if criterion not in cls.FEATURE_GROUPS:
            raise ValueError(f"Unknown criterion '{criterion}'. Available criteria: {list(cls.FEATURE_GROUPS.keys())}")
        
        if criterion == 'all_features':
            # Special case: will be handled dynamically when dataset is available
            return []
        
        return cls.FEATURE_GROUPS[criterion].copy()
    
    @classmethod
    def list_available_criteria(cls) -> Dict[str, List[str]]:
        """
        List all available feature selection criteria and their features.
        
        Returns:
            Dictionary mapping criteria to feature lists
        """
        return {k: v.copy() for k, v in cls.FEATURE_GROUPS.items()}
    
    @classmethod
    def add_custom_feature_group(cls, name: str, features: List[str]):
        """
        Add a custom feature group.
        
        Args:
            name: Name of the feature group
            features: List of features in the group
        """
        if name in cls.FEATURE_GROUPS:
            print(f"Warning: Overwriting existing feature group '{name}'")
        
        cls.FEATURE_GROUPS[name] = features.copy()
        print(f"Added custom feature group '{name}' with {len(features)} features")
    
    def __init__(self, selected_features: Union[List[str], str] = None, 
                 criterion: str = None):
        super().__init__()
        
        # Handle different initialization modes
        if criterion is not None:
            # Criterion-based selection
            self.criterion = criterion
            self.selected_features = None  # Will be resolved when dataset is available
            print(f"Feature selector initialized with criterion '{criterion}'")
            
        elif selected_features is not None:
            # Direct feature list selection
            if isinstance(selected_features, str):
                # Single feature as string
                self.selected_features = [selected_features]
            else:
                # List of features
                self.selected_features = selected_features
            self.criterion = None
            print(f"Feature selector initialized with {len(self.selected_features)} features directly")
            
        else:
            raise ValueError("Either selected_features or criterion must be provided")
        
        self.feature_indices = None
    
    def _resolve_selected_features(self, dataset: Dataset) -> List[str]:
        """Resolve selected features based on criterion or direct selection."""
        if self.criterion is not None:
            if self.criterion == 'all_features':
                # Return all available features
                return dataset.feature_names.copy()
            else:
                # Get predefined feature group and filter against available features
                predefined_features = self.get_feature_group(self.criterion)
                selected_features = [f for f in predefined_features if f in dataset.feature_names]
                
                # Warn about missing features
                missing_features = [f for f in predefined_features if f not in dataset.feature_names]
                if missing_features:
                    print(f"Warning: Some features from '{self.criterion}' group not available: {missing_features}")
                
                return selected_features
        else:
            # Direct feature selection - filter against available features
            selected_features = [f for f in self.selected_features if f in dataset.feature_names]
            
            # Warn about missing features
            missing_features = [f for f in self.selected_features if f not in dataset.feature_names]
            if missing_features:
                print(f"Warning: Some selected features not available in dataset: {missing_features}")
            
            return selected_features
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply feature selection to the dataset."""
        # Resolve selected features based on dataset
        resolved_features = self._resolve_selected_features(dataset)
        
        if len(resolved_features) == 0:
            raise ValueError("No selected features found in dataset")
        
        # Get feature indices
        self.feature_indices = [dataset.feature_names.index(f) for f in resolved_features]
        
        # Create new dataset
        new_dataset = SharpDataset(
            h5_path=dataset.h5_path,
            timesteps=dataset.timesteps,
            label=dataset.label,
            use_mask=dataset.use_mask,
            multi_class=dataset.multi_class
        )
        
        # Copy data with selected features
        new_dataset.X = dataset.X[:, :, self.feature_indices]
        
        new_dataset.Y = dataset.Y
        new_dataset.harp_numbers = dataset.harp_numbers
        new_dataset._raw_timestamps = dataset._raw_timestamps
        new_dataset._raw_labels = dataset._raw_labels
        #new_dataset._raw_labels = dataset._raw_labels
        if dataset.use_mask:
            new_dataset.mask = dataset.mask[:, :, self.feature_indices]
        if dataset.multi_class:
            new_dataset.label_map = dataset.label_map
        
        # Update feature names
        new_dataset.feature_names = [dataset.feature_names[i] for i in self.feature_indices]
        
        print(f"Feature selection complete: {len(dataset.feature_names)} → {len(new_dataset.feature_names)} features")
        return new_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Select specified features from the sample."""
        if self.feature_indices is None:
            raise ValueError("Transform must be called on dataset before transforming samples")
        
        sample['X'] = sample['X'][:, self.feature_indices]
        if 'mask' in sample:
            sample['mask'] = sample['mask'][:, self.feature_indices]
        return sample

class TimeWindowSelector(Transform):
    """Transform to select data within a specified time window.
    
    This filter keeps only samples whose issuing time (last timestamp) falls within
    the specified date range. This is useful for creating temporally consistent
    datasets or focusing on specific time periods.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        verbose: Whether to print detailed filtering information
    """
    
    def __init__(self, start_date: str, end_date: str, verbose: bool = True):
        super().__init__()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.verbose = verbose
        self.valid_indices = None
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply time window selection to the dataset."""
        timestamps = dataset.timestamps
        
        # Get issuing times (last timestamp of each sample)
        issuing_times = [pd.to_datetime(ts[-1]) for ts in timestamps]
        
        # Find samples within the time window
        self.valid_indices = np.array([
            i for i, issuing_time in enumerate(issuing_times)
            if self.start_date <= issuing_time <= self.end_date
        ])
        
        if self.verbose:
            print(f"Time window selection:")
            print(f"  Start date: {self.start_date}")
            print(f"  End date: {self.end_date}")
            print(f"  Original samples: {len(dataset)}")
            print(f"  Selected samples: {len(self.valid_indices)}")
            print(f"  Samples removed: {len(dataset) - len(self.valid_indices)}")
            print(f"  Retention rate: {len(self.valid_indices)/len(dataset)*100:.1f}%")
            
            # Show temporal coverage
            if len(self.valid_indices) > 0:
                filtered_times = [issuing_times[i] for i in self.valid_indices]
                print(f"  Selected time range: {min(filtered_times)} to {max(filtered_times)}")
        
        # Create a new dataset object without reloading from HDF5
        new_dataset = SharpDataset.__new__(SharpDataset)
        
        # Copy all attributes from the original dataset
        for attr in ['h5_path', 'timesteps', 'label', 'use_mask', 'multi_class', 'feature_names']:
            setattr(new_dataset, attr, getattr(dataset, attr))
        
        # Copy filtered data
        new_dataset.X = dataset.X[self.valid_indices]
        new_dataset.Y = dataset.Y[self.valid_indices]
        new_dataset.harp_numbers = dataset.harp_numbers[self.valid_indices]
        new_dataset._raw_timestamps = dataset._raw_timestamps[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        if dataset.use_mask:
            new_dataset.mask = dataset.mask[self.valid_indices]
        if dataset.multi_class:
            new_dataset.label_map = dataset.label_map
        
        return new_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Select sample based on time window."""
        if self.valid_indices is None:
            raise ValueError("Transform must be called on dataset before transforming samples")
        return sample

class TimeWindowFilter(Transform):
    """Transform to filter out data within a specified time window.
    
    This filter removes samples whose issuing time (last timestamp) falls within
    the specified date range. This is useful for removing specific time periods
    that may contain data quality issues, instrument downtime, or other anomalies.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        verbose: Whether to print detailed filtering information
    """
    
    def __init__(self, start_date: str, end_date: str, verbose: bool = True):
        super().__init__()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.verbose = verbose
        self.valid_indices = None
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply time window filtering to the dataset."""
        timestamps = dataset.timestamps
        
        # Get issuing times (last timestamp of each sample)
        issuing_times = [pd.to_datetime(ts[-1]) for ts in timestamps]
        
        # Find samples OUTSIDE the time window (exclude samples within the window)
        self.valid_indices = np.array([
            i for i, issuing_time in enumerate(issuing_times)
            if not self._overlaps_excluded_period(issuing_time)
        ])
        
        if self.verbose:
            print(f"Time window filtering:")
            print(f"  Filter out start date: {self.start_date}")
            print(f"  Filter out end date: {self.end_date}")
            print(f"  Original samples: {len(dataset)}")
            print(f"  Remaining samples: {len(self.valid_indices)}")
            print(f"  Samples filtered out: {len(dataset) - len(self.valid_indices)}")
            print(f"  Retention rate: {len(self.valid_indices)/len(dataset)*100:.1f}%")
            
            # Show temporal coverage
            if len(self.valid_indices) > 0:
                remaining_times = [issuing_times[i] for i in self.valid_indices]
                print(f"  Remaining time range: {min(remaining_times)} to {max(remaining_times)}")
                
                # Check if there are gaps in the temporal coverage
                sorted_times = sorted(remaining_times)
                if len(sorted_times) > 1:
                    # Find the largest gap that might correspond to the excluded window
                    time_gaps = [(sorted_times[i+1] - sorted_times[i]).total_seconds() / 3600 
                                for i in range(len(sorted_times)-1)]
                    max_gap_hours = max(time_gaps)
                    if max_gap_hours > 24:  # Gap larger than 1 day
                        print(f"  Largest temporal gap: {max_gap_hours:.1f} hours")
                        
                        # Add detailed gap information
                        max_gap_idx = time_gaps.index(max_gap_hours)
                        gap_start = sorted_times[max_gap_idx]
                        gap_end = sorted_times[max_gap_idx + 1]
                        print(f"  Gap period: {gap_start} to {gap_end}")
                        print(f"  Expected excluded period: {self.start_date} to {self.end_date}")
                        
                        # Check if the gap matches the excluded period
                        gap_matches = (gap_start <= self.start_date and gap_end >= self.end_date)
                        print(f"  Gap matches excluded period: {gap_matches}")
        
        # Create a new dataset object without reloading from HDF5
        new_dataset = SharpDataset.__new__(SharpDataset)
        
        # Copy all attributes from the original dataset
        for attr in ['h5_path', 'timesteps', 'label', 'use_mask', 'multi_class', 'feature_names']:
            setattr(new_dataset, attr, getattr(dataset, attr))
        
        # Copy remaining data (excluded samples are not included)
        new_dataset.X = dataset.X[self.valid_indices]
        new_dataset.Y = dataset.Y[self.valid_indices]
        new_dataset.harp_numbers = dataset.harp_numbers[self.valid_indices]
        new_dataset._raw_timestamps = dataset._raw_timestamps[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        if dataset.use_mask:
            new_dataset.mask = dataset.mask[self.valid_indices]
        if dataset.multi_class:
            new_dataset.label_map = dataset.label_map
        
        return new_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Transform a single sample."""
        if self.valid_indices is None:
            raise ValueError("Transform must be called on dataset before transforming samples")
        return sample

class TimeWindowExcluder(Transform):
    """Transform to exclude data within a specified time window.
    
    This filter removes samples whose 24-hour window overlaps with the specified date range.
    This is useful for removing specific time periods that may contain data quality issues,
    instrument downtime, or other anomalies.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        verbose: Whether to print detailed filtering information
    """
    
    def __init__(self, start_date: str, end_date: str, verbose: bool = True):
        super().__init__()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.verbose = verbose
        self.valid_indices = None
    
    def _overlaps_excluded_period(self, issuing_time):
        """Check if a sample's 24-hour window overlaps with the excluded period."""
        # Calculate the 24-hour window for this sample
        start_time = issuing_time - pd.Timedelta(hours=24)
        end_time = issuing_time
        
        # Check if either start time or end time falls within the excluded period
        start_overlaps = self.start_date <= start_time <= self.end_date
        end_overlaps = self.start_date <= end_time <= self.end_date
        
        return start_overlaps or end_overlaps
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply time window exclusion to the dataset."""
        timestamps = dataset.timestamps
        
        # Get issuing times (last timestamp of each sample)
        issuing_times = [pd.to_datetime(ts[-1]) for ts in timestamps]
        
        # Find samples OUTSIDE the time window (exclude samples within the window)
        self.valid_indices = np.array([
            i for i, issuing_time in enumerate(issuing_times)
            if not self._overlaps_excluded_period(issuing_time)
        ])
        
        if self.verbose:
            print(f"Time window exclusion:")
            print(f"  Exclude start date: {self.start_date}")
            print(f"  Exclude end date: {self.end_date}")
            print(f"  Original samples: {len(dataset)}")
            print(f"  Remaining samples: {len(self.valid_indices)}")
            print(f"  Samples excluded: {len(dataset) - len(self.valid_indices)}")
            print(f"  Retention rate: {len(self.valid_indices)/len(dataset)*100:.1f}%")
            
            # Show temporal coverage
            if len(self.valid_indices) > 0:
                remaining_times = [issuing_times[i] for i in self.valid_indices]
                print(f"  Remaining time range: {min(remaining_times)} to {max(remaining_times)}")
                
                # Check if there are gaps in the temporal coverage
                sorted_times = sorted(remaining_times)
                if len(sorted_times) > 1:
                    # Find the largest gap that might correspond to the excluded window
                    time_gaps = [(sorted_times[i+1] - sorted_times[i]).total_seconds() / 3600 
                                for i in range(len(sorted_times)-1)]
                    max_gap_hours = max(time_gaps)
                    if max_gap_hours > 24:  # Gap larger than 1 day
                        print(f"  Largest temporal gap: {max_gap_hours:.1f} hours")
                        
                        # Add detailed gap information
                        max_gap_idx = time_gaps.index(max_gap_hours)
                        gap_start = sorted_times[max_gap_idx]
                        gap_end = sorted_times[max_gap_idx + 1]
                        print(f"  Gap period: {gap_start} to {gap_end}")
                        print(f"  Expected excluded period: {self.start_date} to {self.end_date}")
                        
                        # Check if the gap matches the excluded period
                        gap_matches = (gap_start <= self.start_date and gap_end >= self.end_date)
                        print(f"  Gap matches excluded period: {gap_matches}")
        
        # Create a new dataset object without reloading from HDF5
        new_dataset = SharpDataset.__new__(SharpDataset)
        
        # Copy all attributes from the original dataset
        for attr in ['h5_path', 'timesteps', 'label', 'use_mask', 'multi_class', 'feature_names']:
            setattr(new_dataset, attr, getattr(dataset, attr))
        
        # Copy remaining data (excluded samples are not included)
        new_dataset.X = dataset.X[self.valid_indices]
        new_dataset.Y = dataset.Y[self.valid_indices]
        new_dataset.harp_numbers = dataset.harp_numbers[self.valid_indices]
        new_dataset._raw_timestamps = dataset._raw_timestamps[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
#        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        if dataset.use_mask:
            new_dataset.mask = dataset.mask[self.valid_indices]
        if dataset.multi_class:
            new_dataset.label_map = dataset.label_map
        
        return new_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Transform a single sample."""
        if self.valid_indices is None:
            raise ValueError("Transform must be called on dataset before transforming samples")
        return sample

class MissingnessFilter(Transform):
    """Filter data based on maximum allowed ratio of NaN values."""
    
    def __init__(self, max_nan_ratio: float = 0.95):
        super().__init__()
        self.max_nan_ratio = max_nan_ratio
        self.valid_indices = None
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply the transform to a dataset."""
        # Calculate NaN ratio for each sample
        X = dataset.X
        total_elements = X.shape[1] * X.shape[2]  # timesteps * features
        nan_counts = torch.isnan(X).sum(dim=(1, 2))
        nan_ratio = nan_counts / total_elements
        
        # Filter samples
        valid_mask = nan_ratio <= self.max_nan_ratio
        self.valid_indices = torch.where(valid_mask)[0]
        
        # Debug: Check label distribution before filtering
        original_labels = dataset._raw_labels
        unique_labels, counts = np.unique(original_labels, return_counts=True)
        print(f"Original label distribution: {dict(zip(unique_labels, counts))}")
        
        # Create a new dataset object without reloading from HDF5
        new_dataset = SharpDataset.__new__(SharpDataset)
        
        # Copy all attributes from the original dataset
        for attr in ['h5_path', 'timesteps', 'label', 'use_mask', 'multi_class', 'feature_names']:
            setattr(new_dataset, attr, getattr(dataset, attr))
        
        # Copy filtered data
        new_dataset.X = dataset.X[self.valid_indices]
        new_dataset.Y = dataset.Y[self.valid_indices]
        new_dataset.harp_numbers = dataset.harp_numbers[self.valid_indices]
        new_dataset._raw_timestamps = dataset._raw_timestamps[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        if dataset.use_mask:
            new_dataset.mask = dataset.mask[self.valid_indices]
        if dataset.multi_class:
            new_dataset.label_map = dataset.label_map
        
        # Debug: Check label distribution after filtering
        filtered_labels = new_dataset._raw_labels
        unique_labels, counts = np.unique(filtered_labels, return_counts=True)
        print(f"Filtered label distribution: {dict(zip(unique_labels, counts))}")
        
        return new_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Transform a single sample."""
        if self.valid_indices is None:
            raise ValueError("Transform must be called on dataset before transforming samples")
        return sample

class MultiWindowMissingnessFilter(Transform):
    """Filter data based on missingness threshold across multiple time step windows.
    
    This filter ensures dataset consistency across different model configurations
    that use different time step lengths. It checks that each sample has acceptable
    missingness in ALL specified time windows.
    
    Args:
        time_windows: List of time step lengths to check (e.g., [1, 16, 31, 61, 121])
        max_nan_ratio: Maximum allowed ratio of NaN values in each window
        verbose: Whether to print detailed filtering information
    """
    
    def __init__(self, time_windows: List[int], max_nan_ratio: float = 0.95, verbose: bool = True):
        super().__init__()
        self.time_windows = sorted(time_windows)  # Sort for consistent ordering
        self.max_nan_ratio = max_nan_ratio
        self.verbose = verbose
        self.valid_indices = None
        self.filtering_stats = {}
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply the transform to a dataset."""
        X = dataset.X
        n_samples, n_timesteps, n_features = X.shape
        
        if self.verbose:
            print(f"Multi-window missingness filtering:")
            print(f"  Time windows: {self.time_windows}")
            print(f"  Max NaN ratio: {self.max_nan_ratio}")
            print(f"  Dataset shape: {X.shape}")
        
        # Check if dataset has enough timesteps for all windows
        max_window = max(self.time_windows)
        if n_timesteps < max_window:
            raise ValueError(f"Dataset has {n_timesteps} timesteps but largest window is {max_window}")
        
        # Initialize valid mask for all samples
        valid_mask = torch.ones(n_samples, dtype=torch.bool)
        self.filtering_stats = {}
        
        # Check missingness for each time window
        for window_size in self.time_windows:
            if self.verbose:
                print(f"\nChecking window size: {window_size}")
            
            # Take the last 'window_size' timesteps (FIXED: was first, now last)
            X_window = X[:, -window_size:, :]
            
            # Calculate NaN ratio for this window
            total_elements = window_size * n_features
            nan_counts = torch.isnan(X_window).sum(dim=(1, 2))
            nan_ratio = nan_counts / total_elements
            
            # Filter samples for this window
            window_valid_mask = nan_ratio <= self.max_nan_ratio
            
            # Update overall valid mask (sample must be valid in ALL windows)
            valid_mask = valid_mask & window_valid_mask
            
            # Store statistics
            n_valid = window_valid_mask.sum().item()
            n_filtered = (~window_valid_mask).sum().item()
            self.filtering_stats[window_size] = {
                'valid': n_valid,
                'filtered': n_filtered,
                'valid_ratio': n_valid / n_samples
            }
            
            if self.verbose:
                print(f"  Valid samples: {n_valid}/{n_samples} ({n_valid/n_samples*100:.1f}%)")
                print(f"  Filtered samples: {n_filtered}")
        
        # Get final valid indices
        self.valid_indices = torch.where(valid_mask)[0]
        final_n_valid = len(self.valid_indices)
        final_n_filtered = n_samples - final_n_valid
        
        if self.verbose:
            print(f"\nFinal results:")
            print(f"  Total valid samples: {final_n_valid}/{n_samples} ({final_n_valid/n_samples*100:.1f}%)")
            print(f"  Total filtered samples: {final_n_filtered}")
        
        # Debug: Check label distribution before filtering
        original_labels = dataset._raw_labels
        unique_labels, counts = np.unique(original_labels, return_counts=True)
        if self.verbose:
            print(f"Original label distribution: {dict(zip(unique_labels, counts))}")
        
        # Create a new dataset object without reloading from HDF5
        new_dataset = SharpDataset.__new__(SharpDataset)
        
        # Copy all attributes from the original dataset
        for attr in ['h5_path', 'timesteps', 'label', 'use_mask', 'multi_class', 'feature_names']:
            setattr(new_dataset, attr, getattr(dataset, attr))
        
        # Copy filtered data
        new_dataset.X = dataset.X[self.valid_indices]
        new_dataset.Y = dataset.Y[self.valid_indices]
        new_dataset.harp_numbers = dataset.harp_numbers[self.valid_indices]
        new_dataset._raw_timestamps = dataset._raw_timestamps[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        if dataset.use_mask:
            new_dataset.mask = dataset.mask[self.valid_indices]
        if dataset.multi_class:
            new_dataset.label_map = dataset.label_map
        
        # Debug: Check label distribution after filtering
        filtered_labels = new_dataset._raw_labels
        unique_labels, counts = np.unique(filtered_labels, return_counts=True)
        if self.verbose:
            print(f"Filtered label distribution: {dict(zip(unique_labels, counts))}")
        
        return new_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Transform a single sample."""
        if self.valid_indices is None:
            raise ValueError("Transform must be called on dataset before transforming samples")
        return sample
    
    def get_filtering_summary(self) -> Dict:
        """Get a summary of the filtering results."""
        return {
            'time_windows': self.time_windows,
            'max_nan_ratio': self.max_nan_ratio,
            'filtering_stats': self.filtering_stats,
            'total_valid_samples': len(self.valid_indices) if self.valid_indices is not None else 0
        }

class MultiWindowEmptyFilter(Transform):
    """Filter out completely empty samples across multiple time step windows.
    
    This filter removes samples that are completely empty (all NaN) in ANY of the
    specified time windows. It ensures that all samples have at least some valid
    data in ALL time windows.
    
    Args:
        time_windows: List of time step lengths to check (e.g., [1, 16, 31, 61, 121])
        verbose: Whether to print detailed filtering information
    """
    
    def __init__(self, time_windows: List[int], verbose: bool = True):
        super().__init__()
        self.time_windows = sorted(time_windows)  # Sort for consistent ordering
        self.verbose = verbose
        self.valid_indices = None
        self.filtering_stats = {}
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply the transform to a dataset."""
        X = dataset.X
        n_samples, n_timesteps, n_features = X.shape
        
        if self.verbose:
            print(f"Multi-window empty sample filtering:")
            print(f"  Time windows: {self.time_windows}")
            print(f"  Dataset shape: {X.shape}")
        
        # Check if dataset has enough timesteps for all windows
        max_window = max(self.time_windows)
        if n_timesteps < max_window:
            raise ValueError(f"Dataset has {n_timesteps} timesteps but largest window is {max_window}")
        
        # Initialize valid mask for all samples
        valid_mask = torch.ones(n_samples, dtype=torch.bool)
        self.filtering_stats = {}
        
        # Check for completely empty samples in each time window
        for window_size in self.time_windows:
            if self.verbose:
                print(f"\nChecking window size: {window_size}")
            
            # Take the last 'window_size' timesteps (FIXED: was first, now last)
            X_window = X[:, -window_size:, :]
            
            # Check if each sample has any non-NaN values in this window
            # A sample is completely empty if ALL values in the window are NaN
            has_valid_data = ~torch.isnan(X_window).all(dim=(1, 2))
            
            # Update overall valid mask (sample must have valid data in ALL windows)
            valid_mask = valid_mask & has_valid_data
            
            # Store statistics
            n_valid = has_valid_data.sum().item()
            n_filtered = (~has_valid_data).sum().item()
            self.filtering_stats[window_size] = {
                'valid': n_valid,
                'filtered': n_filtered,
                'valid_ratio': n_valid / n_samples,
                'empty_samples': n_filtered
            }
            
            if self.verbose:
                print(f"  Valid samples: {n_valid}/{n_samples} ({n_valid/n_samples*100:.1f}%)")
                print(f"  Completely empty samples: {n_filtered}")
        
        # Get final valid indices
        self.valid_indices = torch.where(valid_mask)[0]
        final_n_valid = len(self.valid_indices)
        final_n_filtered = n_samples - final_n_valid
        
        if self.verbose:
            print(f"\nFinal results:")
            print(f"  Total valid samples: {final_n_valid}/{n_samples} ({final_n_valid/n_samples*100:.1f}%)")
            print(f"  Total filtered samples: {final_n_filtered}")
        
        # Debug: Check label distribution before filtering
        original_labels = dataset._raw_labels
        unique_labels, counts = np.unique(original_labels, return_counts=True)
        if self.verbose:
            print(f"Original label distribution: {dict(zip(unique_labels, counts))}")
        
        # Create a new dataset object without reloading from HDF5
        new_dataset = SharpDataset.__new__(SharpDataset)
        
        # Copy all attributes from the original dataset
        for attr in ['h5_path', 'timesteps', 'label', 'use_mask', 'multi_class', 'feature_names']:
            setattr(new_dataset, attr, getattr(dataset, attr))
        
        # Copy filtered data
        new_dataset.X = dataset.X[self.valid_indices]
        new_dataset.Y = dataset.Y[self.valid_indices]
        new_dataset.harp_numbers = dataset.harp_numbers[self.valid_indices]
        new_dataset._raw_timestamps = dataset._raw_timestamps[self.valid_indices]
        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
#        new_dataset._raw_labels = dataset._raw_labels[self.valid_indices]
        if dataset.use_mask:
            new_dataset.mask = dataset.mask[self.valid_indices]
        if dataset.multi_class:
            new_dataset.label_map = dataset.label_map
        
        # Debug: Check label distribution after filtering
        filtered_labels = new_dataset._raw_labels
        unique_labels, counts = np.unique(filtered_labels, return_counts=True)
        if self.verbose:
            print(f"Filtered label distribution: {dict(zip(unique_labels, counts))}")
        
        return new_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Transform a single sample."""
        if self.valid_indices is None:
            raise ValueError("Transform must be called on dataset before transforming samples")
        return sample
    
    def get_filtering_summary(self) -> Dict:
        """Get a summary of the filtering results."""
        return {
            'time_windows': self.time_windows,
            'filtering_stats': self.filtering_stats,
            'total_valid_samples': len(self.valid_indices) if self.valid_indices is not None else 0
        }


class CMASKFilter(Transform): #TODO: Update the CMask filtering - this function is still not fully tested 
    """
    Transform to filter events based on CMASK values across multiple timesteps.
    
    Args:
        threshold: CMASK threshold value for filtering
        strategy: Filtering strategy ('any', 'majority', 'average', 'minimum')
        majority_threshold: Threshold for majority strategy (default: 0.5)
        verbose: Whether to print filtering statistics
    """
    
    def __init__(self, threshold: float, strategy: str = 'any', majority_threshold: float = 0.5, verbose: bool = True):
        self.threshold = threshold
        self.strategy = strategy
        self.majority_threshold = majority_threshold
        self.verbose = verbose
        self.filtering_summary = {}
        
    def filter_events(self, X_data: np.ndarray, cmask_idx: int) -> np.ndarray:
        """
        Filter events based on CMASK values using vectorized operations.
        Args:
            X_data: Array of shape [n_events, n_timesteps, n_features]
            cmask_idx: Index of CMASK feature in the feature array
        Returns:
            mask: Boolean array of shape [n_events] indicating which events to keep
        """
        cmask_values = X_data[:, :, cmask_idx]
        valid_mask = ~np.isnan(cmask_values)
        
        if self.strategy == 'average':
            mean_cmask = np.nanmean(cmask_values, axis=1)
            mask = mean_cmask >= self.threshold
        elif self.strategy == 'any':
            mask = np.all((cmask_values >= self.threshold) | ~valid_mask, axis=1)
        elif self.strategy == 'majority':
            low_quality = (cmask_values < self.threshold) & valid_mask
            low_quality_count = np.sum(low_quality, axis=1)
            valid_count = np.sum(valid_mask, axis=1)
            valid_count = np.maximum(valid_count, 1)
            low_quality_ratio = low_quality_count / valid_count
            mask = low_quality_ratio <= self.majority_threshold
        elif self.strategy == 'minimum':
            min_cmask = np.nanmin(cmask_values, axis=1)
            mask = min_cmask >= self.threshold
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Handle events with all NaN values
        all_nan_events = np.all(~valid_mask, axis=1)
        mask[all_nan_events] = False
        
        # For statistics
        mean_cmask = np.nanmean(cmask_values, axis=1)
        self._last_mean_cmask = mean_cmask
        return mask
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply the transform to the dataset."""
        if not isinstance(dataset, SharpDataset):
            raise ValueError("CMASKFilter only works with SharpDataset")
        try:
            cmask_idx = dataset.feature_names.index('CMASK')
        except ValueError:
            raise ValueError("CMASK feature not found in dataset")
        if self.verbose:
            print(f"CMASK found at index {cmask_idx}")
            print(f"Applying {self.strategy} filter with threshold {self.threshold}")
        mask = self.filter_events(dataset.X.numpy(), cmask_idx)
        n_original = len(dataset.X)
        n_filtered = np.sum(mask)
        n_removed = n_original - n_filtered
        if self.verbose:
            print(f"Original events: {n_original}")
            print(f"Events after filtering: {n_filtered}")
            print(f"Events removed: {n_removed} ({n_removed/n_original*100:.1f}%)")
            # CMASK statistics
            cmask_values = dataset.X.numpy()[:, :, cmask_idx]
            print(f"\nCMASK Statistics:")
            print(f"CMASK range: {np.nanmin(cmask_values)} to {np.nanmax(cmask_values)}")
            print(f"CMASK mean: {np.nanmean(cmask_values):.2f}")
            print(f"CMASK median: {np.nanmedian(cmask_values):.2f}")
            mean_cmask = self._last_mean_cmask
            print(f"Events with mean CMASK < {self.threshold}: {np.sum(mean_cmask < self.threshold)}")
            print(f"Events with mean CMASK >= {self.threshold}: {np.sum(mean_cmask >= self.threshold)}")
        self.filtering_summary = {
            'cmask_filter_threshold': self.threshold,
            'cmask_filter_strategy': self.strategy,
            'original_events': n_original,
            'filtered_events': n_filtered,
            'removed_events': n_removed,
            'removal_percentage': n_removed/n_original*100
        }
        if not mask.all():
            dataset.X = dataset.X[mask]
            dataset.mask = dataset.mask[mask]
            dataset.Y = dataset.Y[mask]
            dataset.harp_numbers = dataset.harp_numbers[mask]
            dataset._raw_timestamps = dataset._raw_timestamps[mask]
            dataset._raw_labels = dataset._raw_labels[mask]
            dataset._raw_labels = dataset._raw_labels[mask]
        return dataset
    
    def transform_sample(self, sample: dict) -> dict:
        raise NotImplementedError("CMASKFilter does not support single sample transformation")
    
    def get_filtering_summary(self) -> dict:
        return self.filtering_summary

class Compose(Transform):
    """Compose multiple transforms together.
    
    Args:
        transforms: List of transforms to apply in sequence
    """
    
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms
    
    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply all transforms to the dataset."""
        processed_dataset = dataset
        for transform in self.transforms:
            processed_dataset = transform(processed_dataset)
        return processed_dataset
    
    def transform_sample(self, sample: dict) -> dict:
        """Apply all transforms to the sample."""
        for transform in self.transforms:
            sample = transform.transform_sample(sample)
        return sample

class SharpDatasetSplitter:   #TODO: Double check the date logic. I updated the splitting logic to be aligned with the GropedStratifiedKfoldSplit function. There could be issues wiyth dates handling.
    """Splitter class for creating train/validation/test splits with temporal awareness.
    
    This class provides methods to split datasets while maintaining temporal separation
    and preventing data leakage between train, validation, and test sets.
    
    Attributes:
        strategy: Splitting strategy ('chronological' or 'random')
        test_size: Proportion of data to use for testing
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        temporal_distr_viz: Whether to plot temporal distributions
        label_distr_viz: Whether to plot label distributions
        output_dir: Directory to save plots and files
        force_separation: Whether to drop overlapping samples between train/val and test sets
    """
    
    def __init__(
        self,
        strategy: str = 'chronological',
        test_size: float = 0.15,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        temporal_distr_viz: bool = False,
        label_distr_viz: bool = False,
        output_dir: str = ".",
        force_separation: bool = True
    ):
        """Initialize the splitter.
        
        Args:
            strategy: Splitting strategy ('chronological' or 'random')
            test_size: Proportion of data to use for testing
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            temporal_distr_viz: Whether to plot temporal distributions
            label_distr_viz: Whether to plot label distributions
            output_dir: Directory to save plots and files
            force_separation: Whether to drop overlapping samples between train/val and test sets
        """
        self.strategy = strategy
        self.test_size = test_size
        self.n_folds = n_folds
        self.random_state = random_state
        self.temporal_distr_viz = temporal_distr_viz
        self.label_distr_viz = label_distr_viz
        self.output_dir = output_dir
        self.force_separation = force_separation
        
        # Track splitting method for filename generation
        self.splitting_method = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_splitting_method_name(self, test_time_window=None, test_cutoff_date=None):
        """Generate a descriptive name for the splitting method used."""
        base_name = ""
        if self.strategy == 'chronological':
            if test_time_window:
                start_date, end_date = test_time_window
                base_name = f"chrono_window_{start_date}_to_{end_date}"
            elif test_cutoff_date:
                base_name = f"chrono_cutoff_{test_cutoff_date}"
            else:
                base_name = f"chrono_default_{int(self.test_size*100)}pct"
        else:
            base_name = f"random_{int(self.test_size*100)}pct"
        
        # Add force_separation information
        if self.strategy == 'chronological':
            if self.force_separation:
                base_name += "_strict"
            else:
                base_name += "_overlap"
        
        return base_name
    
    def _get_plot_filename(self, plot_type, splitting_method=None):
        """Generate a filename for plots with splitting method information."""
        if splitting_method is None:
            splitting_method = self.splitting_method or "unknown"
        
        # Clean the splitting method name for filename use
        clean_method = splitting_method.replace('/', '_').replace(':', '_')
        
        if plot_type == "temporal":
            return f"temporal_distribution_{clean_method}.png"
        elif plot_type == "label":
            return f"label_distribution_{clean_method}.png"
        elif plot_type == "label_normalized":
            return f"label_distribution_normalized_{clean_method}.png"
        else:
            return f"{plot_type}_{clean_method}.png"
    
    def plot_splits(self, splits: Dict[str, List[Dataset]], title: str = "Temporal Distribution of Splits", verbose: bool = False) -> None:
        """Plot the temporal distribution of train, validation, and test splits.
        
        Args:
            splits: Dictionary containing train, val, and test datasets
            title: Title for the plot
            verbose: Whether to print verbose output
        """
        import matplotlib
        # Set the backend to a non-interactive one if not in a notebook
        #matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:green'}
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Cache for storing decoded timestamps
        timestamp_cache = {}
        has_data = False
        
        for split in ['train', 'val', 'test']:
            split_datasets = splits[split]
            all_times = []
            
            for dataset in split_datasets:
                if isinstance(dataset, Subset):
                    # Get the original dataset and indices
                    original_dataset = dataset.dataset
                    indices = dataset.indices
                    
                    # Use cached timestamps if available
                    if id(original_dataset) not in timestamp_cache:
                        # Decode timestamps once and cache them
                        raw_timestamps = original_dataset._raw_timestamps
                        decoded_times = []
                        for ts_sample in raw_timestamps:
                            try:
                                # Get the last timestamp and decode it
                                last_ts = ts_sample[-1]
                                decoded_ts = original_dataset.decode_if_bytes(last_ts).replace(" ", "T")
                                dt = datetime.fromisoformat(decoded_ts)
                                decoded_times.append(dt)
                            except Exception as e:
                                continue
                        
                        timestamp_cache[id(original_dataset)] = np.array(decoded_times)
                    
                    # Get the cached timestamps for the current indices
                    timestamps = timestamp_cache[id(original_dataset)][indices]
                else:
                    # For regular datasets, use the timestamps property
                    timestamps = np.array([ts[-1] for ts in dataset.timestamps])
                
                all_times.extend(timestamps)
            
            if all_times:  # Only plot if we have times
                ax.hist(all_times, bins=100, alpha=0.5, 
                       label=f"{split.upper()} ({len(all_times)} samples)", 
                       color=colors[split])
                has_data = True
        
        if not has_data:  # Check if any data was plotted
            if verbose:
                print("No valid data to plot!")
            plt.close(fig)
            return
        
        ax.set_title(title)
        ax.set_xlabel("Issuing Time")
        ax.set_ylabel("Frequency")
        ax.legend()
        
    
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Save the plot to a file
        output_path = os.path.join(self.output_dir, self._get_plot_filename("temporal"))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\nPlot saved to {output_path}")
        
        # Try to display the plot (only in interactive environments)
        try:
            # Check if we're in an interactive environment
            if matplotlib.get_backend() in ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg']:
                plt.show()
            else:
                if verbose:
                    print(f"Non-interactive backend detected. Plot saved to {output_path}")
        except Exception as e:
            if verbose:
                print(f"Could not display plot interactively: {e}")
                print(f"Plot has been saved to {output_path}")
        
        plt.close(fig)
    
    def plot_label_distributions(self, splits: Dict[str, List[Dataset]], title: str = "Label Distribution Across Splits", verbose: bool = False) -> None:
        """Plot the label distribution for train, validation, and test splits.
        
        Args:
            splits: Dictionary containing train, val, and test datasets
            title: Title for the plot
            verbose: Whether to print verbose output
        """
        import matplotlib.pyplot as plt
        from collections import Counter
        
        def decode_label(label):
            if isinstance(label, (bytes, np.bytes_)):
                return label.decode("utf-8")
            if hasattr(label, "item"):  # torch tensor or numpy scalar
                return decode_label(label.item())
            return str(label)
        
        n_folds = len(splits['train'])
        total_plots = n_folds + 1  # +1 for test set
        n_cols = 2
        n_rows = int(np.ceil(total_plots / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
        axes = axes.flatten()
        
        # Process each fold
        for i in range(n_folds):
            train_labels, val_labels = [], []
            
            # Get train labels
            train_dataset = splits['train'][i]
            if isinstance(train_dataset, Subset):
                train_labels = [decode_label(label) for label in train_dataset.dataset._raw_labels[train_dataset.indices]]
            else:
                train_labels = [decode_label(label) for label in train_dataset._raw_labels]
            
            # Get val labels
            val_dataset = splits['val'][i]
            if isinstance(val_dataset, Subset):
                val_labels = [decode_label(label) for label in val_dataset.dataset._raw_labels[val_dataset.indices]]
            else:
                val_labels = [decode_label(label) for label in val_dataset._raw_labels]
            
            # Get unique labels and counts
            all_labels = sorted(set(train_labels + val_labels))
            x = np.arange(len(all_labels))
            bar_width = 0.4
            
            train_counts = Counter(train_labels)
            val_counts = Counter(val_labels)
            
            train_values = [train_counts.get(lbl, 0) for lbl in all_labels]
            val_values = [val_counts.get(lbl, 0) for lbl in all_labels]
            
            # Plot
            ax = axes[i]
            ax.bar(x - bar_width/2, train_values, width=bar_width, label='Train', color='skyblue')
            ax.bar(x + bar_width/2, val_values, width=bar_width, label='Validation', color='orange')
            
            ax.set_yscale("log")
            ax.set_title(f"Fold {i+1}")
            ax.set_xticks(x)
            ax.set_xticklabels(all_labels, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot test set
        test_labels = []
        test_dataset = splits['test'][0]
        
        if isinstance(test_dataset, Subset):
            test_labels = [decode_label(label) for label in test_dataset.dataset._raw_labels[test_dataset.indices]]
        else:
            test_labels = [decode_label(label) for label in test_dataset._raw_labels]
        
        test_counts = Counter(test_labels)
        labels = sorted(test_counts.keys())
        test_values = [test_counts.get(lbl, 0) for lbl in labels]
        x = np.arange(len(labels))
        
        ax = axes[n_folds]
        ax.bar(x, test_values, width=0.6, color='mediumseagreen')
        ax.set_yscale("log")
        ax.set_title("Test Set")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(total_plots, len(axes)):
            axes[j].axis("off")
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, self._get_plot_filename("label"))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_normalized_label_distributions(self, splits: Dict[str, List[Dataset]], original_dataset: Dataset = None, title: str = "Label Distribution Deviation from Original", verbose: bool = False) -> None:
        """Plot the deviation of label distribution percentages from the original dataset.
        
        This plot shows the difference between split percentages and original dataset percentages,
        making it easy to spot which splits have proportion imbalances.
        
        Args:
            splits: Dictionary containing train, val, and test datasets
            original_dataset: Original dataset to use as reference (optional)
            title: Title for the plot
            verbose: Whether to print verbose output
        """
        import matplotlib.pyplot as plt
        from collections import Counter
        
        def decode_label(label):
            if isinstance(label, (bytes, np.bytes_)):
                return label.decode("utf-8")
            if hasattr(label, "item"):  # torch tensor or numpy scalar
                return decode_label(label.item())
            return str(label)
        
        # Get original dataset distribution if provided, otherwise use first train fold
        if original_dataset is not None:
            original_labels = [decode_label(label) for label in original_dataset._raw_labels]
            original_counts = Counter(original_labels)
            original_total = len(original_labels)
        else:
            # Use first train fold as reference
            first_train_dataset = splits['train'][0]
            if isinstance(first_train_dataset, Subset):
                original_labels = [decode_label(label) for label in first_train_dataset.dataset._raw_labels[first_train_dataset.indices]]
            else:
                original_labels = [decode_label(label) for label in first_train_dataset._raw_labels]
            original_counts = Counter(original_labels)
            original_total = len(original_labels)
        
        # Calculate original percentages
        original_percentages = {label: (count / original_total) * 100 for label, count in original_counts.items()}
        
        n_folds = len(splits['train'])
        total_plots = n_folds + 1  # +1 for test set
        n_cols = 2
        n_rows = int(np.ceil(total_plots / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
        axes = axes.flatten()
        
        # Get all unique labels across all splits
        all_unique_labels = set()
        for split_name in ['train', 'val', 'test']:
            for dataset in splits[split_name]:
                if isinstance(dataset, Subset):
                    labels = [decode_label(label) for label in dataset.dataset._raw_labels[dataset.indices]]
                else:
                    labels = [decode_label(label) for label in dataset._raw_labels]
                all_unique_labels.update(labels)
        
        all_labels = sorted(all_unique_labels)
        x = np.arange(len(all_labels))
        bar_width = 0.4
        
        # Process each fold
        for i in range(n_folds):
            train_labels, val_labels = [], []
            
            # Get train labels
            train_dataset = splits['train'][i]
            if isinstance(train_dataset, Subset):
                train_labels = [decode_label(label) for label in train_dataset.dataset._raw_labels[train_dataset.indices]]
            else:
                train_labels = [decode_label(label) for label in train_dataset._raw_labels]
            
            # Get val labels
            val_dataset = splits['val'][i]
            if isinstance(val_dataset, Subset):
                val_labels = [decode_label(label) for label in val_dataset.dataset._raw_labels[val_dataset.indices]]
            else:
                val_labels = [decode_label(label) for label in val_dataset._raw_labels]
            
            # Calculate percentages and deviations
            train_counts = Counter(train_labels)
            val_counts = Counter(val_labels)
            train_total = len(train_labels)
            val_total = len(val_labels)
            
            train_percentages = [train_counts.get(lbl, 0) / train_total * 100 if train_total > 0 else 0 for lbl in all_labels]
            val_percentages = [val_counts.get(lbl, 0) / val_total * 100 if val_total > 0 else 0 for lbl in all_labels]
            
            # Calculate deviations from original
            original_values = [original_percentages.get(lbl, 0) for lbl in all_labels]
            train_deviations = [train_pct - orig_pct for train_pct, orig_pct in zip(train_percentages, original_values)]
            val_deviations = [val_pct - orig_pct for val_pct, orig_pct in zip(val_percentages, original_values)]
            
            # Plot deviations
            ax = axes[i]
            bars1 = ax.bar(x - bar_width/2, train_deviations, width=bar_width, label='Train', color='skyblue', alpha=0.8)
            bars2 = ax.bar(x + bar_width/2, val_deviations, width=bar_width, label='Validation', color='orange', alpha=0.8)
            
            # Add zero line (perfect balance)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
            
            # Add tolerance bands (±1%, ±2%, ±5%)
            ax.axhline(y=1, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.axhline(y=-1, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.axhline(y=2, color='orange', linestyle=':', alpha=0.3, linewidth=1)
            ax.axhline(y=-2, color='orange', linestyle=':', alpha=0.3, linewidth=1)
            
            ax.set_title(f"Fold {i+1} (Train: {train_total}, Val: {val_total})")
            ax.set_xticks(x)
            ax.set_xticklabels(all_labels, rotation=45)
            ax.set_ylabel("Deviation from Original (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add deviation values on bars
            for bar in bars1:
                height = bar.get_height()
                if abs(height) > 0.1:  # Only show if deviation is significant
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                           f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                if abs(height) > 0.1:  # Only show if deviation is significant
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                           f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # Plot test set
        test_labels = []
        test_dataset = splits['test'][0]
        
        if isinstance(test_dataset, Subset):
            test_labels = [decode_label(label) for label in test_dataset.dataset._raw_labels[test_dataset.indices]]
        else:
            test_labels = [decode_label(label) for label in test_dataset._raw_labels]
        
        test_counts = Counter(test_labels)
        test_total = len(test_labels)
        test_percentages = [test_counts.get(lbl, 0) / test_total * 100 if test_total > 0 else 0 for lbl in all_labels]
        
        # Calculate test deviations
        test_deviations = [test_pct - orig_pct for test_pct, orig_pct in zip(test_percentages, original_values)]
        
        ax = axes[n_folds]
        bars3 = ax.bar(x, test_deviations, width=0.6, color='mediumseagreen', alpha=0.8)
        
        # Add zero line and tolerance bands
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
        ax.axhline(y=1, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=-1, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=2, color='orange', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(y=-2, color='orange', linestyle=':', alpha=0.3, linewidth=1)
        
        ax.set_title(f"Test Set ({test_total} samples)")
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=45)
        ax.set_ylabel("Deviation from Original (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add deviation values on bars
        for bar in bars3:
            height = bar.get_height()
            if abs(height) > 0.1:  # Only show if deviation is significant
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                       f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # Hide unused subplots
        for j in range(total_plots, len(axes)):
            axes[j].axis("off")
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, self._get_plot_filename("label_normalized"))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Close the figure to free memory
        plt.close(fig)
        
        if verbose:
            print(f"Label distribution deviation plot saved to {output_path}")
            print(f"Original dataset percentages: {original_percentages}")
            print("Deviation interpretation:")
            print("  • 0% = Perfect balance (same as original)")
            print("  • +X% = Over-represented (X% more than original)")
            print("  • -X% = Under-represented (X% less than original)")
            print("  • Red dotted lines = ±1% tolerance")
            print("  • Orange dotted lines = ±2% tolerance")
        if verbose:
            print(f"\nLabel distribution plot saved to {output_path}")
        
        # Try to display (only in interactive environments)
        try:
            # Check if we're in an interactive environment
            import matplotlib
            if matplotlib.get_backend() in ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg']:
                plt.show()
            else:
                if verbose:
                    print(f"Non-interactive backend detected. Plot saved to {output_path}")
        except Exception as e:
            if verbose:
                print(f"Could not display plot interactively: {e}")
                print(f"Plot has been saved to {output_path}")
        
        plt.close(fig)
    
    def analyze_splits(self, splits: Dict[str, List[Dataset]], original_dataset: Dataset = None, verbose: bool = True) -> None:
        """Analyze the created splits and print detailed statistics.
        
        Args:
            splits: Dictionary containing train, val, and test datasets
            original_dataset: Original dataset before splitting (for reference)
            verbose: Whether to print detailed analysis
        """
        if verbose:
            print("\n=== Split Analysis ===")
        
        # Get test dataset (same for all folds)
        test_ds = splits['test'][0]
        
        # Get first fold's train/val as representative
        train_ds = splits['train'][0]
        val_ds = splits['val'][0]
        
        # Get unique HARPs
        train_harps = set(train_ds.dataset.harp_numbers[train_ds.indices])
        val_harps = set(val_ds.dataset.harp_numbers[val_ds.indices])
        test_harps = set(test_ds.dataset.harp_numbers[test_ds.indices])
        
        # Check HARP overlap
        train_val_overlap = len(train_harps.intersection(val_harps))
        train_test_overlap = len(train_harps.intersection(test_harps))
        val_test_overlap = len(val_harps.intersection(test_harps))
        
        if verbose:
            print("\nHARP Statistics:")
            print(f"  Train HARPs: {len(train_harps)}")
            print(f"  Val HARPs: {len(val_harps)}")
            print(f"  Test HARPs: {len(test_harps)}")
            print(f"  Train-Val overlap: {train_val_overlap} HARPs")
            print(f"  Train-Test overlap: {train_test_overlap} HARPs")
            print(f"  Val-Test overlap: {val_test_overlap} HARPs")
        
        # Get time ranges for each split
        def get_time_ranges(dataset):
            if len(dataset) == 0:
                return None, None
            timestamps = dataset.dataset.timestamps[dataset.indices]
            start_times = [pd.to_datetime(ts[0]) for ts in timestamps]
            end_times = [pd.to_datetime(ts[-1]) for ts in timestamps]
            return min(start_times), max(end_times)
        
        train_start, train_end = get_time_ranges(train_ds)
        val_start, val_end = get_time_ranges(val_ds)
        test_start, test_end = get_time_ranges(test_ds)
        
        if verbose:
            print("\nTime Range Coverage:")
            print(f"  Train: {train_start} to {train_end}")
            print(f"  Val: {val_start} to {val_end}")
            if test_start is not None and test_end is not None:
                print(f"  Test: {test_start} to {test_end}")
            else:
                print("  Test: No data")
            
            # Check for test set separation
            print("\nTest Set Separation Analysis:")
            if test_start is not None:
                # Get the latest start time from train and val sets
                latest_train_val_start = max(
                    train_start if train_start is not None else pd.Timestamp.min,
                    val_start if val_start is not None else pd.Timestamp.min
                )
                
                if latest_train_val_start >= test_start:
                    print("  Warning: Train/Val sets start after test set starts")
                    print(f"    Latest Train/Val start: {latest_train_val_start}")
                    print(f"    Test starts at: {test_start}")
        
        # Analyze label distribution
        def get_label_dist(dataset):
            if len(dataset) == 0:
                return {}
            labels = dataset.dataset._raw_labels[dataset.indices]  # Use decomposed labels for consistency
            unique_labels, counts = np.unique(labels, return_counts=True)
            return dict(zip(unique_labels, counts))
        
        train_dist = get_label_dist(train_ds)
        val_dist = get_label_dist(val_ds)
        test_dist = get_label_dist(test_ds)
        
        if verbose:
            print("\nLabel Distribution:")
            print("  Train:")
            for label, count in train_dist.items():
                print(f"    {label}: {count} samples ({count/len(train_ds)*100:.2f}%)")
            print("  Val:")
            for label, count in val_dist.items():
                print(f"    {label}: {count} samples ({count/len(val_ds)*100:.2f}%)")
            print("  Test:")
            if test_dist:
                for label, count in test_dist.items():
                    print(f"    {label}: {count} samples ({count/len(test_ds)*100:.2f}%)")
            else:
                print("    No data")
            
            # Print sample counts
            print("\nSample Counts:")
            print(f"  Train: {len(train_ds)} samples")
            print(f"  Val: {len(val_ds)} samples")
            print(f"  Test: {len(test_ds)} samples")
        
        # Add visualizations if enabled
        if self.temporal_distr_viz:
            self.plot_splits(splits, verbose=verbose)
        
        if self.label_distr_viz:
            self.plot_label_distributions(splits, verbose=verbose)
            # Also generate normalized label distribution plot
            self.plot_normalized_label_distributions(splits, original_dataset=original_dataset, verbose=verbose)
        
        if verbose:
            print("\n" + "="*50)
    
    def split(
        self,
        dataset: Dataset,
        test_time_window: Optional[Tuple[str, str]] = None,
        test_cutoff_date: Optional[str] = None,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[Dataset]]:
        """Split the dataset into train, validation, and test sets.
        
        Args:
            dataset: Dataset to split
            test_time_window: Optional tuple of (start_date, end_date) for test set
            test_cutoff_date: Optional cutoff date for test set
            output_path: Optional path to save splits (overrides output_dir)
            verbose: Whether to print detailed analysis
            
        Returns:
            Dictionary containing lists of train, validation, and test datasets
        """
        # Update output directory if output_path is provided
        if output_path is not None:
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
        
        # Set the splitting method name for plot filenames
        self.splitting_method = self._get_splitting_method_name(test_time_window, test_cutoff_date)
        
        # Store verbose state for internal methods
        self._verbose = verbose
        
        splits = self._chronological_split(dataset, test_time_window, test_cutoff_date) if self.strategy == 'chronological' else self._random_split(dataset)
        
        # Analyze the splits only if verbose
        if verbose:
            self.analyze_splits(splits, original_dataset=dataset, verbose=verbose)
        
        return splits
    
    def _chronological_split(
        self,
        dataset: Dataset,
        test_time_window: Optional[Tuple[str, str]] = None,
        test_cutoff_date: Optional[str] = None
    ) -> Dict[str, List[Dataset]]:
        """Split dataset chronologically, ensuring no temporal overlap between splits.
        
        Args:
            dataset: Dataset to split
            test_time_window: Optional tuple of (start_date, end_date) for test set
            test_cutoff_date: Optional cutoff date for test set
        """
        # Create DataFrame with metadata
        df = pd.DataFrame({
            'index': np.arange(len(dataset)),
            'harp': dataset.harp_numbers,
            'label': dataset._raw_labels,  # Use decomposed labels for stratification
            'start_time': [pd.to_datetime(ts[0]) for ts in dataset.timestamps],
            'end_time': [pd.to_datetime(ts[-1]) for ts in dataset.timestamps]
        })
        
        # Sort all data chronologically by start time
        df = df.sort_values('start_time')
        
        if test_time_window:
            # Split based on time window
            start = pd.to_datetime(test_time_window[0]).normalize()
            end = pd.to_datetime(test_time_window[1]).normalize() #+ pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            # Find samples that start and end within the test window
            test_mask = (df['start_time'] > start) & (df['end_time'] < end)
            test_df = df[test_mask]
            
            # Find samples that end before the test window starts OR start after it ends
            train_val_mask = (df['end_time'] < start) | (df['start_time'] > end)
            train_val_df = df[train_val_mask]
            
            if len(test_df) == 0:
                print(f"\nWarning: No HARPs fully within the test window {start} to {end}")
                print("Consider using a different time window.")
                return None
            
        elif test_cutoff_date:
            # Split based on cutoff date
            cutoff = pd.to_datetime(test_cutoff_date).normalize()
            
            # HARP-LEVEL SPLITTING: Group by HARP and determine entire HARP assignment
            # This prevents HARPs from being split across train/test
            
            # Get unique HARPs and their temporal ranges
            harp_ranges = df.groupby('harp').agg({
                'start_time': 'min',
                'end_time': 'max'
            }).reset_index()
            
            # Assign HARPs to train/val or test based on their temporal range
            # Train/Val: HARPs that end BEFORE the cutoff date
            train_val_harps = harp_ranges[harp_ranges['end_time'] < cutoff]['harp'].tolist()
            
            # Test: HARPs that start AFTER the cutoff date (exclusive)
            test_harps = harp_ranges[harp_ranges['start_time'] > cutoff]['harp'].tolist()
            
            # Apply HARP-level masks to the original dataframe
            train_val_mask = df['harp'].isin(train_val_harps)
            test_mask = df['harp'].isin(test_harps)
            
            train_val_df = df[train_val_mask]
            test_df = df[test_mask]
            
            if len(test_df) == 0:
                print(f"\nWarning: No HARPs start after the cutoff date {cutoff}")
                print("This might indicate that the cutoff date is too recent.")
                print("Consider using an earlier cutoff date or a different splitting strategy.")
                return None
        
        else:
            # Default chronological split: use the last test_size proportion as test set
            # Sort by end_time to ensure temporal separation
            df = df.sort_values('end_time')
            
            # Calculate the cutoff index for test set based on test_size percentage
            test_size_int = max(1, int(len(df) * self.test_size))
            
            # Find the cutoff date: the end_time of the (len - test_size_int)th sample
            cutoff_idx = len(df) - test_size_int
            cutoff_date = df.iloc[cutoff_idx]['end_time']
            
            # HARP-LEVEL SPLITTING: Group by HARP and determine entire HARP assignment
            # This prevents HARPs from being split across train/test
            
            # Get unique HARPs and their temporal ranges
            harp_ranges = df.groupby('harp').agg({
                'start_time': 'min',
                'end_time': 'max'
            }).reset_index()
            
            # Assign HARPs to train/val or test based on their temporal range
            # Train/Val: HARPs that end BEFORE the cutoff date
            train_val_harps = harp_ranges[harp_ranges['end_time'] < cutoff_date]['harp'].tolist()
            
            # Test: HARPs that start AFTER the cutoff date (exclusive)
            test_harps = harp_ranges[harp_ranges['start_time'] > cutoff_date]['harp'].tolist()
            
            # Apply HARP-level masks to the original dataframe
            train_val_mask = df['harp'].isin(train_val_harps)
            test_mask = df['harp'].isin(test_harps)
            
            train_val_df = df[train_val_mask]
            test_df = df[test_mask]
            
            if len(test_df) == 0:
                print(f"\nWarning: No HARPs start after the calculated cutoff date {cutoff_date}")
                return None
        
        # Find overlapping HARPs (HARPs that don't fall into either train_val or test)
        # With HARP-level splitting, this should be empty, but we check for safety
        overlapping_mask = ~(train_val_mask | test_mask)
        overlapping_df = df[overlapping_mask]
        n_overlapping = len(overlapping_df)
        
        # Also check for HARP-level overlap between train/val and test sets
        train_val_harps = set(train_val_df['harp'].unique())
        test_harps = set(test_df['harp'].unique())
        harp_overlap = train_val_harps.intersection(test_harps)
        n_harp_overlap = len(harp_overlap)
        
        # Debug information
        if self._verbose:
            print(f"\n Overlap Analysis:")
            print(f"  Total unique HARPs: {len(df['harp'].unique())}")
            print(f"  Train/Val unique HARPs: {len(train_val_df['harp'].unique())}")
            print(f"  Test unique HARPs: {len(test_df['harp'].unique())}")
            print(f"  Calculated overlapping: {n_overlapping}")
            print(f"  HARP overlap between train/val and test: {n_harp_overlap}")
            if n_harp_overlap > 0:
                print(f"  Overlapping HARPs: {sorted(harp_overlap)}")
        
        # Show temporal ranges for debugging
        if len(train_val_df) > 0 and len(test_df) > 0:
            train_val_start = train_val_df['start_time'].min()
            train_val_end = train_val_df['end_time'].max()
            test_start = test_df['start_time'].min()
            test_end = test_df['end_time'].max()
            
            if self._verbose:
                print(f"\n Temporal Ranges:")
                print(f"  Train/Val: {train_val_start} to {train_val_end}")
                print(f"  Test: {test_start} to {test_end}")
                
                # For time window method, check if the split is working as intended
                if test_time_window:
                    # In time window method, train/val should be before and after the test window
                    # Check if train/val spans the entire dataset range (indicating proper split)
                    dataset_start = df['start_time'].min()
                    dataset_end = df['end_time'].max()
                    
                    if train_val_start <= dataset_start and train_val_end >= dataset_end:
                        print(f"  Time window split successful: Train/Val covers periods before and after test window")
                        print(f"  Split strategy: Test window ({test_start} to {test_end}) with train/val from surrounding periods")
                        
                        # Show the separate train/val time windows
                        train_val_before = train_val_df[train_val_df['end_time'] <= test_start]
                        train_val_after = train_val_df[train_val_df['start_time'] >= test_end]
                        
                        if len(train_val_before) > 0:
                            before_start = train_val_before['start_time'].min()
                            before_end = train_val_before['end_time'].max()
                            print(f"  Train/Val (before): {before_start} to {before_end} ({len(train_val_before['harp'].unique())} unique HARPs)")
                        
                        if len(train_val_after) > 0:
                            after_start = train_val_after['start_time'].min()
                            after_end = train_val_after['end_time'].max()
                            print(f"  Train/Val (after): {after_start} to {after_end} ({len(train_val_after['harp'].unique())} unique HARPs)")
                    else:
                        print(f"  Time window split may not be optimal")
                        print(f"  Dataset range: {dataset_start} to {dataset_end}")
                        
                elif test_cutoff_date:
                    # For cutoff date method, check for actual temporal overlap
                    temporal_overlap = (train_val_end >= test_start) and (test_end >= train_val_start)
                    if temporal_overlap:
                        print(f"  WARNING: Temporal overlap detected!")
                        overlap_start = max(train_val_start, test_start)
                        overlap_end = min(train_val_end, test_end)
                        print(f"  Overlap period: {overlap_start} to {overlap_end}")
                    else:
                        print(f"  No temporal overlap detected")
                    
                else:
                    # For default chronological split, show the split information
                    print(f"  Default chronological split: Test set contains the most recent {len(test_df['harp'].unique())} unique HARPs")
                    print(f"  Train/Val set contains the remaining {len(train_val_df['harp'].unique())} unique HARPs")
                    print(f"  Temporal separation: Test set ends at {test_end}, Train/Val set starts at {train_val_start}")
        
        if n_overlapping > 0:
            if self.force_separation:
                if self._verbose:
                    print(f"Dropped {n_overlapping} overlapping samples to maintain strict separation")
                # The overlapping samples are already excluded by the mask logic above
                # No additional action needed - they're not in train_val_df or test_df
            else:
                if self._verbose:
                    print(f"Found {n_overlapping} overlapping samples - adding them to test set")
                # Add overlapping samples to test set only (to avoid data leakage)
                test_df = pd.concat([test_df, overlapping_df], ignore_index=True)
        else:
            if self._verbose:
                print(f"No overlapping samples found - clean temporal separation achieved")
        
        # Handle HARP-level overlap
        if n_harp_overlap > 0:
            if self.force_separation:
                if self._verbose:
                    print(f"WARNING: {n_harp_overlap} HARPs overlap between train/val and test sets!")
                    print(f"This indicates a problem with the HARP-level splitting logic.")
                    print(f"Overlapping HARPs: {sorted(harp_overlap)}")
                # For now, we'll keep the current split but warn the user
                # In a future version, we could implement more sophisticated HARP assignment
            else:
                if self._verbose:
                    print(f"Found {n_harp_overlap} overlapping HARPs - this is expected with event-level splitting")
        else:
            if self._verbose:
                print(f"No HARP overlap detected - clean HARP-level separation achieved")
        
        # Create train/val splits using stratification
        train_datasets, val_datasets = [], []
        
        # Use StratifiedGroupKFold for train/val splits
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Create train/val splits
        for train_idx, val_idx in sgkf.split(train_val_df['harp'], train_val_df['label'], groups=train_val_df['harp']):
            train_df = train_val_df.iloc[train_idx]
            val_df = train_val_df.iloc[val_idx]
            
            # Create datasets
            train_indices = train_df['index'].tolist()
            val_indices = val_df['index'].tolist()
            
            train_datasets.append(Subset(dataset, train_indices))
            val_datasets.append(Subset(dataset, val_indices))
        
        # Create test dataset
        test_indices = test_df['index'].tolist()
        test_dataset = Subset(dataset, test_indices)
        
        return {
            'train': train_datasets,
            'val': val_datasets,
            'test': [test_dataset]
        }
    
    def _random_split(self, dataset: Dataset) -> Dict[str, List[Dataset]]:
        """Split dataset randomly using stratified sampling."""
        # Create DataFrame with metadata
        df = pd.DataFrame({
            'index': np.arange(len(dataset)),
            'harp': dataset.harp_numbers,
            'label': dataset._raw_labels  # Use decomposed labels for stratification
        })
        
        # Split into train/val and test
        sgkf = StratifiedGroupKFold(
            n_splits=int(1/self.test_size),
            shuffle=True,
            random_state=self.random_state
        )
        
        for train_val_idx, test_idx in sgkf.split(df['harp'], df['label'], groups=df['harp']):
            train_val_df = df.iloc[train_val_idx]
            test_df = df.iloc[test_idx]
            break
        
        # Create train/val splits using stratified k-fold
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        train_datasets, val_datasets = [], []
        for train_idx, val_idx in sgkf.split(train_val_df['harp'], train_val_df['label'], groups=train_val_df['harp']):
            train_indices = train_val_df.iloc[train_idx]['index'].tolist()
            val_indices = train_val_df.iloc[val_idx]['index'].tolist()
            
            train_datasets.append(Subset(dataset, train_indices))
            val_datasets.append(Subset(dataset, val_indices))
        
        # Create test dataset
        test_indices = test_df['index'].tolist()
        test_dataset = Subset(dataset, test_indices)
        
        return {
            'train': train_datasets,
            'val': val_datasets,
            'test': [test_dataset]
        }

def save_split_to_h5(dataset: Dataset, output_path: str, timesteps: int = 121, verbose: bool = False) -> None:
    """Save a dataset split to HDF5 format.
    
    Args:
        dataset: The dataset to save (can be a Subset or regular Dataset)
        output_path: Path where to save the HDF5 file
        timesteps: Number of timesteps to keep (if dataset has more timesteps than requested)
        verbose: Whether to print detailed statistics
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Handle Subset objects
    if hasattr(dataset, 'indices'):
        # This is a Subset, extract the original dataset and indices
        original_dataset = dataset.dataset
        indices = dataset.indices
        
        # Extract data using indices
        X = original_dataset.X[indices].numpy()
        harp_numbers = original_dataset.harp_numbers[indices]
        raw_timestamps = original_dataset._raw_timestamps[indices]
        raw_labels = original_dataset._raw_labels[indices]
        raw_labels = original_dataset._raw_labels[indices]
        feature_names = original_dataset.feature_names
        
        # Truncate to requested timesteps if needed
        if timesteps < X.shape[1]:
            X = X[:, -timesteps:, :]
            raw_timestamps = raw_timestamps[:, -timesteps:]
    else:
        # This is a regular Dataset
        X = dataset.X.numpy()
        harp_numbers = dataset.harp_numbers
        raw_timestamps = dataset._raw_timestamps
        raw_labels = dataset._raw_labels
        raw_labels = dataset._raw_labels
        feature_names = dataset.feature_names
        
        # Truncate to requested timesteps if needed
        if timesteps < X.shape[1]:
            X = X[:, -timesteps:, :]
            raw_timestamps = raw_timestamps[:, -timesteps:]
    
    with h5py.File(output_path, 'w') as hf:
        # Save features X
        hf.create_dataset('X', data=X)
        
        # Save HARPNUM
        hf.create_dataset('HARPNUM', data=harp_numbers)
        
        # Save timestamps
        hf.create_dataset('timestamps', data=raw_timestamps)
        
        # Save labels
        hf.create_dataset('Y', data=np.array(raw_labels, dtype='S'))
        hf.create_dataset('Y_u', data=np.array(raw_labels, dtype='S'))
        
        # Save feature names if available
        if feature_names:
            feature_names_str = ','.join(feature_names)
            hf.attrs['feature_names'] = feature_names_str
        
        # Print dataset statistics only if verbose
        if verbose:
            print(f"Split saved to {output_path}")
            print(f"  - Samples: {len(X)}")
            print(f"  - Features: {X.shape[2]}")
            print(f"  - Timesteps: {X.shape[1]}")
            print(f"  - Labels: {np.unique(raw_labels)}")
            print(f"  - Decomposed Labels: {np.unique(raw_labels)}")

def save_splits(splits: Dict[str, List[Dataset]], output_dir: str, prefix: str = "flare_split", output_path: Optional[str] = None, splitting_method: Optional[str] = None, verbose: bool = False) -> None:
    """Save all splits to HDF5 files.
    
    Args:
        splits: Dictionary containing train, val, and test datasets
        output_dir: Directory to save the splits
        prefix: Prefix for the output files
        output_path: Optional specific path to save splits (overrides output_dir)
        splitting_method: Optional splitting method name to include in filenames
        verbose: Whether to print detailed statistics for each split
    """
    # Use output_path if provided, otherwise use output_dir
    if output_path is not None:
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
        prefix = os.path.splitext(os.path.basename(output_path))[0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Include splitting method in prefix if provided
    if splitting_method:
        # Clean the splitting method name for filename use
        clean_method = splitting_method.replace('/', '_').replace(':', '_')
        prefix = f"{prefix}_{clean_method}"
    
    # Save test set
    test_subset = splits['test'][0]
    test_path = os.path.join(output_dir, f"{prefix}_test.h5")
    save_split_to_h5(test_subset, test_path, verbose=verbose)
    
    # Save train and val sets for each fold
    for fold in range(len(splits['train'])):
        # Save train set
        train_subset = splits['train'][fold]
        train_path = os.path.join(output_dir, f"{prefix}_cv_{fold}_train.h5")
        save_split_to_h5(train_subset, train_path, verbose=verbose)
        
        # Save val set
        val_subset = splits['val'][fold]
        val_path = os.path.join(output_dir, f"{prefix}_cv_{fold}_val.h5")
        save_split_to_h5(val_subset, val_path, verbose=verbose)
    
    if verbose:
        print(f"All splits saved to {output_dir}") 