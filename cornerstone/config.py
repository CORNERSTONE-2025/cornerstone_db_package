import torch
from dataset_config import AVAILABLE_DATASETS

# Get the specific dataset configuration
CURRENT_DATASET = AVAILABLE_DATASETS["M+_17f"]



# Task configuration - from specific dataset
TASK_CONFIG = {
    "name": CURRENT_DATASET["name"],
    "description": CURRENT_DATASET["description"],
    "data_prefix": CURRENT_DATASET["data_prefix"]
}
 

'''# Model hyperparameters - from specific dataset
MODEL_CONFIG = {
    "input_dim": CURRENT_DATASET["input_dim"],
    "model_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout_prob": 0.3,
}





}

#INCEPTIONTIME
MODEL_CONFIG = {
    "input_size": CURRENT_DATASET["input_dim"]
}




#XGBOOST
MODEL_CONFIG={
    "n_estimators" : 100,
    "max_depth" : 3,
    "learning_rate" : 0.1, 
    "mask_concat" : True 
}



#MINIROCKET
MODEL_CONFIG = {
    "num_kernels": 500,  
    #"random_state": 42 # during debugging makes sense to fix it to 42 while for the final version set it as None
}

#ROCKET
MODEL_CONFIG = {
    "num_kernels": 5000,  
    #"random_state": 42 # during debugging makes sense to fix it to 42 while for the final version set it as None
}

'''

# Current model selection
CURRENT_MODEL = "MINIROCKET" 

# Training hyperparameters
TRAINING_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "patience": 20,
    "n_folds": 5,
    "lr_scheduler": {
        "type": "CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 1e-6
    },
    "threshold_optimization": {
        "enabled": True,
        "metric": "TSS",
        "n_thresholds": 50,
        "threshold_range": [0.0, 1.0],
        "min_recall": 0.6,
        "min_precision": 0.3,
        "optimize_every_epoch": True,
    },
    "calibration": {
        "enabled": True,
        "kind": "auto",
        "few_pos_threshold": 25,
        "n_bins": 10
    },
    "early_stopping": {
        "monitor": "TSS",
        "mode": "max",
        "min_delta": 1e-3,
        "patience": 20
    },
    "loss_function": {
        "type": "bce",
        "curriculum": {
            "alpha": 0.5,
            "gamma": 2.0,
            "warmup_epochs": 5,
            "transition_epochs": 10,
            "eps": 1e-6,
        },
        "focal": {
            "alpha": 0.96,
            "gamma": 2.0,
            "eps": 1e-6,
        },
        "asymmetric_focal": {
            "alpha": 0.96,
            "gamma_pos": 5.0,
            "gamma_neg": 2.0,
            "eps": 1e-6,
        }
    }
}


# Data configuration - from specific dataset
DATA_CONFIG = {
    "window_map": {  
        "3h": 16,
        "6h": 31,
        "12h": 61,
        "24h": 121,
    },
    "window_descriptions": {
        "ss": "Single Snapshot", 
        "3h": "3-Hour Window",
        "6h": "6-Hour Window",
        "12h": "12-Hour Window",
        "24h": "24-Hour Window",
    },
    "base_path": CURRENT_DATASET["base_path"]
}
'''
# Data configuration - from specific dataset
DATA_CONFIG = {
    "window_map": {
        "ss": 1,
        "3h": 16,
        "6h": 31,
        "12h": 61,
        "24h": 121,
    },
    "window_descriptions": {
        "ss": "Single Snapshot",
        "3h": "3-Hour Window",
        "6h": "6-Hour Window",
        "12h": "12-Hour Window",
        "24h": "24-Hour Window",
    },
    "base_path": CURRENT_DATASET["base_path"]
}
'''
# Device configuration
DEVICE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu" 
}

# Output configuration
OUTPUT_CONFIG = {
    "results_dir": "./results",
    "save_models": True,
    "plot_metrics": ["TSS", "HSS", "F1", "MCC"],
    "plot_training_curves": True, #DP
    "plot_metric_curves": True, #DP
    "plot_boxplot": True, #DP
    "plot_test_metrics": True, #DP
}

def get_model_config():
    """Returns the appropriate model configuration based on CURRENT_MODEL."""
    if CURRENT_MODEL == "LSTM":
        return LSTM_CONFIG
    return MODEL_CONFIG
