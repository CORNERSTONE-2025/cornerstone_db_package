# Dataset-specific configuration
# This file contains only the parameters that change between different datasets
# All training parameters remain in config.py

# Current dataset configuration
CURRENT_DATASET = {
    "name": "C+",  # Task name: C+, M+, X+
    "description": "C-class flares and above",  # Task description
    "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_chrono_default_20pct",  # Dataset folder name
    "input_dim": 17,  # Number of features (17 or 22)
    "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_chrono_default_20pct/splits"  # Path to splits folder
}

# Available datasets (for reference)
AVAILABLE_DATASETS = {
    # # Chrono datasets (existing)
    # "C+_17f": {
    #     "name": "C+",
    #     "description": "C-class flares and above",
    #     "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_chrono_default_20pct",
    #     "input_dim": 17,
    #     "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_chrono_default_20pct/splits"
    # },
    # "C+_22f": {
    #     "name": "C+",
    #     "description": "C-class flares and above", 
    #     "data_prefix": "sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_chrono_default_20pct",
    #     "input_dim": 22,
    #     "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_chrono_default_20pct/splits"
    # },
    # "M+_17f": {
    #     "name": "M+",
    #     "description": "M-class flares and above",
    #     "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_chrono_default_20pct",
    #     "input_dim": 17,
    #     "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_chrono_default_20pct/splits"
    # },
    # "M+_22f": {
    #     "name": "M+",
    #     "description": "M-class flares and above",
    #     "data_prefix": "sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_chrono_default_20pct",
    #     "input_dim": 22,
    #     "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_chrono_default_20pct/splits"
    # },
    # "X+_17f": {
    #     "name": "X+",
    #     "description": "X-class flares and above",
    #     "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_chrono_default_20pct",
    #     "input_dim": 17,
    #     "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_chrono_default_20pct/splits"
    # },
    # "X+_22f": {
    #     "name": "X+",
    #     "description": "X-class flares and above",
    #     "data_prefix": "sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_chrono_default_20pct",
    #     "input_dim": 22,
    #     "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_chrono_default_20pct/splits"
    # },
    
    # Random datasets (newly added)
    "C+_17f_random": {
        "name": "C+",
        "description": "C-class flares and above (random split)",
        "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_random_20pct",
        "input_dim": 17,
        "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_random_20pct/splits"
    },
    "M+_17f": {
    "name": "M+",
    "description": "M-class flares and above",
    "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_chrono_default_20pct",
    "input_dim": 17,
    "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_chrono_default_20pct/splits"
    },
    "C+_22f_random": {
        "name": "C+",
        "description": "C-class flares and above (random split)", 
        "data_prefix": "sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_random_20pct",
        "input_dim": 22,
        "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_C+_random_20pct/splits"
    },
    "M+_17f_random": {
        "name": "M+",
        "description": "M-class flares and above (random split)",
        "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_random_20pct",
        "input_dim": 17,
        "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_random_20pct/splits"
    },
    "M+_22f_random": {
        "name": "M+",
        "description": "M-class flares and above (random split)",
        "data_prefix": "sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_random_20pct",
        "input_dim": 22,
        "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_random_20pct/splits"
    },
    "X+_17f_random": {
        "name": "X+",
        "description": "X-class flares and above (random split)",
        "data_prefix": "sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_random_20pct",
        "input_dim": 17,
        "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_random_20pct/splits"
    },
    "X+_22f_random": {
        "name": "X+",
        "description": "X-class flares and above (random split)",
        "data_prefix": "sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_random_20pct",
        "input_dim": 22,
        "base_path": "./data/dataset_splits/sharp_dataset_db_SHARPC_all_NRT_direct_22features_multiwin_empty_win1-16-31-61-121_no_cmask_X+_random_20pct/splits"
    }
}

def get_dataset_config(dataset_key: str = None):
    """Get dataset configuration by key or return current dataset."""
    if dataset_key and dataset_key in AVAILABLE_DATASETS:
        return AVAILABLE_DATASETS[dataset_key]
    return CURRENT_DATASET

def list_available_datasets():
    """List all available dataset configurations."""
    print("Available datasets:")
    for key, config in AVAILABLE_DATASETS.items():
        print(f"  {key}: {config['name']} ({config['input_dim']} features) - {config['description']}")
        print(f"    Path: {config['base_path']}")
        print()

def set_current_dataset(dataset_key: str):
    """Set the current dataset configuration."""
    if dataset_key not in AVAILABLE_DATASETS:
        print(f"ERROR: Dataset '{dataset_key}' not found")
        return False
    
    global CURRENT_DATASET
    CURRENT_DATASET.clear()  # Clear the existing dict
    CURRENT_DATASET.update(AVAILABLE_DATASETS[dataset_key])  # Update with new values
    
    print(f"SUCCESS: Set current dataset to {dataset_key}")
    print(f"  Name: {CURRENT_DATASET['name']}")
    print(f"  Features: {CURRENT_DATASET['input_dim']}")
    print(f"  Path: {CURRENT_DATASET['base_path']}")
    return True
