import os
from typing import List, Optional
from .config import OUTPUT_CONFIG, TASK_CONFIG, DATA_CONFIG #, CURRENT_MODEL

class PathManager:
    """Manages all path operations for the project."""
    
    def __init__(self, task_name: Optional[str] = None, model_type: Optional[str] = None, dataset_name: Optional[str] = None, base_dir: Optional[str] = None, run_name: Optional[str] = None, data_config: Optional[dict] = None):
        self.task_name = task_name or TASK_CONFIG["name"]
        # Map model types to their file naming convention
        model_type_mapping = {
            "TRANSFORMER": "transformer",
            "TRANSFORMER_ATTENTION": "transformer",
            "LSTM": "lstm",
            "TimesNet": "timesnet",
            "Autoformer": "autoformer",
            "DLinear": "dlinear",
            "ETSformer": "etsformer",
            "FEDformer": "fedformer",
            "LightTS": "lightts",
            "Nonstationary_Transformer": "nonstationary_transformer",
            "Pyraformer": "pyraformer",
            "Reformer": "reformer",
            "Informer": "informer",
            "INCEPTIONTIME": "inceptiontime",
            "ROCKET": "rocket",
            "ONEFITSALL" : "onefitsall"
        } #DP metti tutti i modelli 

        print(model_type)
        raw_model_type = model_type #or CURRENT_MODEL
        self.model_type = model_type_mapping.get(raw_model_type, raw_model_type.lower())
        self.dataset_name = dataset_name or TASK_CONFIG["data_prefix"]
        self.base_dir = base_dir or OUTPUT_CONFIG["results_dir"]
        self.run_name = run_name
        self.data_config = data_config or DATA_CONFIG
        
        # For ablation runs, use run_name as primary folder structure
        if self.run_name and "ablation" in self.base_dir:
            # Structure: results/ablation/run_name/
            self.task_dir = os.path.join(self.base_dir, self.run_name)
        else:
            # Structure: results/dataset_name/task_name/
            self.task_dir = os.path.join(self.base_dir, self.dataset_name, self.task_name)
        
    def ensure_dir(self, *subdirs: str) -> str:
        """Ensures a directory exists and returns its path."""
        path = os.path.join(self.task_dir, *subdirs)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_model_path(
        self,
        window: str,
        fold: int | None = None,
        best: bool = False,
        model_type: str = "PT"
    ) -> str:
        """
        Returns the path to a trained model.
        
        - If fold is provided -> use fold
        - If fold is None -> require best=True, otherwise error
        """
        
        # --- Safety logic ---
        if fold is None:
            if not best:
                raise ValueError(
                    "Either provide a fold number or set best=True "
                    "when requesting a model path."
                )
        else:
            # If fold is set, best must be False
            if best:
                raise ValueError(
                    "Cannot specify both fold and best=True. "
                    "Choose only one."
                )

        # --- Select extension ---
        ext = "pt" if model_type == "PT" else "pkl"

        # --- Build filename ---
        if best:
            filename = f"{self.task_name}_{self.model_type}_model_{window}_best.{ext}"
        else:
            filename = f"{self.task_name}_{self.model_type}_model_{window}_fold{fold}.{ext}"

        # --- Final path ---
        return os.path.join(self.task_dir, "models", filename)
    
        
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
        return os.path.join(self.data_config["base_path"], f"{data_prefix}_{split}.h5")
    
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
        if self.run_name and "ablation" in self.base_dir:
            return self.task_dir
        else:
            return f"results/{self.dataset_name}/{self.task_name}/"
    
    def get_calibration_path(self, filename: str) -> str:
        """Get path for calibration artifacts."""
        calibration_dir = self.ensure_dir("calibration")
        return os.path.join(calibration_dir, filename)
    
    def get_logs_path(self, filename: str) -> str:
        """Get path for log files."""
        logs_dir = self.ensure_dir("logs")
        return os.path.join(logs_dir, filename)
    
    def get_ensemble_path(self, filename: str) -> str:
        """Get path for ensemble evaluation results."""
        ensemble_dir = self.ensure_dir("ensemble", "metrics")
        return os.path.join(ensemble_dir, filename)
    
    def get_ensemble_plots_path(self, filename: str) -> str:
        """Get path for ensemble plots."""
        ensemble_plots_dir = self.ensure_dir("ensemble", "plots")
        return os.path.join(ensemble_plots_dir, filename) 