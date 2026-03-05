import os
from typing import List, Optional
from config import OUTPUT_CONFIG, TASK_CONFIG, CURRENT_MODEL, DATA_CONFIG

class PathManager:
    """Manages all path operations for the project."""
    
    def __init__(self, task_name: Optional[str] = None, model_type: Optional[str] = None, dataset_name: Optional[str] = None):
        self.task_name = task_name or TASK_CONFIG["name"]
        self.model_type = model_type or CURRENT_MODEL.lower()
        self.dataset_name = dataset_name or TASK_CONFIG["data_prefix"]
        self.base_dir = OUTPUT_CONFIG["results_dir"]
        
        # Structure: results/dataset_name/task_name/
        self.task_dir = os.path.join(self.base_dir, self.dataset_name, self.task_name)
        
    def ensure_dir(self, *subdirs: str) -> str:
        """Ensures a directory exists and returns its path."""
        path = os.path.join(self.task_dir, *subdirs)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_model_path(self, window: str, fold: int) -> str:
        """Gets the path for a model file."""
        return os.path.join(
            self.task_dir, 
            "models", 
            f"{self.task_name}_{self.model_type}_model_{window}_fold{fold}.pt"
        )
    
    def get_metrics_path(self, filename: str) -> str:
        """Gets the path for a metrics file."""
        # Add model type to filename if not already present
        if self.model_type not in filename:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{self.model_type}{ext}"
        return os.path.join(self.task_dir, "metrics", filename)
    
    def get_plot_path(self, filename: str) -> str:
        """Gets the path for a plot file."""
        # Add model type to filename if not already present
        if self.model_type not in filename:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{self.model_type}{ext}"
        return os.path.join(self.task_dir, "plots", filename)
    
    def get_attention_path(self, filename: str) -> str:
        """Gets the path for attention map files."""
        # Add model type to filename if not already present
        if self.model_type not in filename:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{self.model_type}{ext}"
        return os.path.join(self.task_dir, "attention_maps", filename)
    
    def get_config_path(self, filename: str) -> str:
        """Gets the path for configuration files."""
        # Add model type to filename if not already present
        if self.model_type not in filename:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{self.model_type}{ext}"
        return os.path.join(self.task_dir, "configs", filename)
    
    def get_data_path(self, data_prefix: str, split: str) -> str:
        """Gets the path for a data file."""
        return os.path.join(DATA_CONFIG["base_path"], f"{data_prefix}_{split}.h5")
    
    def list_models(self, window: str) -> List[str]:
        """Lists all model files for a given window."""
        models_dir = os.path.join(self.task_dir, "models")
        pattern = f"{self.task_name}_{self.model_type}_model_{window}_fold*.pt"
        return [f for f in os.listdir(models_dir) if f.startswith(pattern)]
    
    def get_model_type(self) -> str:
        """Returns the current model type."""
        return self.model_type
    
    def get_task_name(self) -> str:
        """Returns the current task name."""
        return self.task_name
    
    def get_dataset_name(self) -> str:
        """Returns the current dataset name."""
        return self.dataset_name
    
    def get_results_structure(self) -> str:
        """Returns the current results directory structure."""
        return f"results/{self.dataset_name}/{self.task_name}/" 