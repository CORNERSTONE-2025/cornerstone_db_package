import json
import torch
import os
from .dataset_config import AVAILABLE_DATASETS
from ..model_configs import MODEL_CONFIGS
from ..regularization_configs import REGULARIZATION_PARAMS

# default path (can be overridden dynamically)
_config_path = "trakkin_config.json"
_cfg_json = None  # will store the loaded JSON

# ----------------------------
# Config path management
# ----------------------------
def set_config_path(path: str):
    """Set JSON config path dynamically."""
    global _config_path, _cfg_json
    _config_path = path
    _cfg_json = None  # reset, will reload on next access

def load_json_config():
    """Load JSON config lazily from current path."""
    global _config_path, _cfg_json
    if _cfg_json is not None:
        return _cfg_json  # already loaded
    if _config_path is None:
        raise ValueError("Config path not set. Call set_config_path(path) before loading.")
    if not os.path.exists(_config_path):
        raise FileNotFoundError(f"Config file not found: {_config_path}")
    with open(_config_path, "r") as f:
        _cfg_json = json.load(f)
    return _cfg_json

def get_cfg_json():
    return load_json_config()

# ----------------------------
# Lazy-loaded accessors
# ----------------------------
def CURRENT_DATASET():
    cfg = get_cfg_json()
    return AVAILABLE_DATASETS[cfg.get("CURRENT_DATASET", "M+_17f_random")]

def CURRENT_MODELS():
    cfg = get_cfg_json()
    return cfg.get("CURRENT_MODELS", "")

def DATA_CONFIG():
    cfg = get_cfg_json()
    data = cfg.get("DATA_CONFIG", {})
    data["base_path"] = CURRENT_DATASET()["base_path"]
    return data

def TRAINING_CONFIG():
    cfg = get_cfg_json()
    return cfg.get("TRAINING_CONFIG", {})

def OUTPUT_CONFIG():
    cfg = get_cfg_json()
    return cfg.get("OUTPUT_CONFIG", {
        "results_dir": "./results",
        "save_models": True,
        "plot_metrics": ["TSS", "HSS", "F1", "MCC"],
        "plot_training_curves": True,
        "plot_metric_curves": True,
        "plot_boxplot": True,
        "plot_test_metrics": True,
    })

# ----------------------------
# Device config
# ----------------------------
DEVICE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ----------------------------
# Task info
# ----------------------------
def TASK_CONFIG():
    dataset = CURRENT_DATASET()
    return {
        "name": dataset["name"],
        "description": dataset["description"],
        "data_prefix": dataset["data_prefix"]
    }

# ----------------------------
# Model configuration helper
# ----------------------------
def get_model_config(CURRENT_MODEL, overrides=None):
    """
    Returns a model configuration dictionary.
    - Auto-fills enc_in / dec_in based on CURRENT_DATASET input_dim
    - Applies optional overrides dictionary
    """
    base_config = MODEL_CONFIGS.get(CURRENT_MODEL, {}).copy()
    input_dim = CURRENT_DATASET()["input_dim"]

    if input_dim is not None:
        if "enc_in" in base_config and base_config["enc_in"] is None:
            base_config["enc_in"] = input_dim
        if "dec_in" in base_config and base_config["dec_in"] is None:
            base_config["dec_in"] = input_dim
        if "input_size" in base_config and base_config["input_size"] is None:
            base_config["input_size"] = input_dim
        if "input_dim" in base_config and base_config["input_dim"] is None:
            base_config["input_dim"] = input_dim 
        

    cfg = get_cfg_json()
    json_overrides = cfg.get("MODEL_CONFIG_OVERRIDES", {}).get(CURRENT_MODEL, {})
    model_overrides = overrides or {}
    base_config.update(json_overrides)
    base_config.update(model_overrides)

    return base_config


def get_training_config(CURRENT_MODEL ):
    """
    Returns a training configuration dictionary for a given model.
    - Automatically sets default regularization parameters if available
    - Applies optional overrides from the JSON config or passed dictionary
    """
    cfg = get_cfg_json()
    base_training_config = cfg.get("TRAINING_CONFIG", {}).copy()

    # Add default regularization params for this model if available
    if CURRENT_MODEL in REGULARIZATION_PARAMS:
        base_training_config["regularization_params"] = REGULARIZATION_PARAMS[CURRENT_MODEL]["regularization_params"]

    # Apply overrides from JSON
    json_overrides = cfg.get("TRAINING_CONFIG_OVERRIDES", {}).get(CURRENT_MODEL, {})
    base_training_config.update(json_overrides)
 

    return base_training_config
