import json
import os
from datetime import datetime
from typing import Dict, Any 
from .config import MODEL_CONFIGS, TRAINING_CONFIG, DATA_CONFIG, DEVICE_CONFIG, OUTPUT_CONFIG, TASK_CONFIG, CURRENT_DATASET #CURRENT_MODEL, 

class ConfigManager:
    """Manages saving and loading of configuration files for reproducibility."""
    
    def __init__(self, path_manager):
        self.path_manager = path_manager
     
    def save_config(self, current_model, filename: str = None, config_obj=None) -> str: 
        """
        Save current configuration to a text file for later reference.
        
        Args:
            filename: Optional custom filename, otherwise auto-generated
            current_model: Override for CURRENT_MODEL if provided
            task_config: Override for TASK_CONFIG if provided
            
        Returns:
            Path to the saved configuration file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            model_name = current_model 
            task_name = TASK_CONFIG()['name']
            filename = f"{task_name}_config_{timestamp}_{model_name.lower()}.txt"
        
        config_path = self.path_manager.get_config_path(filename)
        
        # Use provided config object or fall back to global configs
        if config_obj:
            config_data = {
                "timestamp": datetime.now().isoformat(), 
                "model_type": current_model,#config_obj.CURRENT_MODEL,
                "task_config": config_obj.TASK_CONFIG,
                "model_config": config_obj.MODEL_CONFIGS.get(current_model, {}), 
                "training_config": config_obj.TRAINING_CONFIG,
                "data_config": config_obj.DATA_CONFIG,
                "device_config": config_obj.DEVICE_CONFIG,
                "output_config": config_obj.OUTPUT_CONFIG
            }
        else:
            # Fall back to global configs
            config_data = {
                "timestamp": datetime.now().isoformat(), 
                "model_type": current_model,
                "task_config": TASK_CONFIG(),
                "model_config": MODEL_CONFIGS.get(current_model, {}), 
                "training_config": TRAINING_CONFIG(),
                "data_config": DATA_CONFIG(),
                "device_config": DEVICE_CONFIG,
                "output_config": OUTPUT_CONFIG()
            }
        
        # Save as formatted text file
        with open(config_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SOLAR FLARE FORECASTING - CONFIGURATION FILE\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {config_data['timestamp']}\n")
            f.write(f"Model Type: {config_data['model_type']}\n")
            f.write(f"Task: {config_data['task_config']()['name']}\n")
            f.write(f"Dataset: {config_data['task_config']()['data_prefix']}\n")
            f.write("\n")
            
            # Task Configuration
            f.write("TASK CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            for key, value in config_data['task_config']().items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Model Configuration
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            for key, value in config_data['model_config'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Training Configuration
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            for key, value in config_data['training_config']().items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Data Configuration
            f.write("DATA CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            for key, value in config_data['data_config']().items():
                if key == "window_map":
                    f.write(f"{key}:\n")
                    for window, steps in value.items():
                        f.write(f"  {window}: {steps} steps\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Device Configuration
            f.write("DEVICE CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            for key, value in config_data['device_config'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Output Configuration
            f.write("OUTPUT CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            for key, value in config_data['output_config']().items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF CONFIGURATION\n")
            f.write("=" * 80 + "\n")
        
        print(f"📝 Configuration saved to: {config_path}")
        return config_path
     
    def save_config_json(self, current_model, filename: str = None, config_obj=None) -> str: 
        """
        Save current configuration as JSON for programmatic access.
        
        Args:
            filename: Optional custom filename, otherwise auto-generated
            current_model: Override for CURRENT_MODEL if provided
            task_config: Override for TASK_CONFIG if provided
            
        Returns:
            Path to the saved JSON configuration file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            model_name = current_model 
            task_name = TASK_CONFIG()['name']
            filename = f"{task_name}_config_{timestamp}_{model_name.lower()}.json"
        
        config_path = self.path_manager.get_config_path(filename)
        
        # Use provided config object or fall back to global configs
        if config_obj:
            config_data = {
                "timestamp": datetime.now().isoformat(), 
                "model_type": current_model,#config_obj.CURRENT_MODEL,
                "task_config": config_obj.TASK_CONFIG,
                "model_config": config_obj.MODEL_CONFIGS.get(current_model, {}), 
                "training_config": config_obj.TRAINING_CONFIG,
                "data_config": config_obj.DATA_CONFIG,
                "device_config": config_obj.DEVICE_CONFIG,
                "output_config": config_obj.OUTPUT_CONFIG
            }
        else:
            # Fall back to global configs
            config_data = {
                "timestamp": datetime.now().isoformat(), 
                "model_type": current_model,
                "task_config": TASK_CONFIG,
                "model_config": MODEL_CONFIGS.get(current_model, {}), 
                "training_config": TRAINING_CONFIG,
                "data_config": DATA_CONFIG,
                "device_config": DEVICE_CONFIG,
                "output_config": OUTPUT_CONFIG
            }
        
        # Save as JSON
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        print(f"📝 JSON configuration saved to: {config_path}")
        return config_path
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Dictionary containing the loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            print(f"📝 Configuration loaded from: {config_path}")
            return config_data
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            return {}
    
    def get_config_summary(self, current_dataset=None, current_model=None, data_config=None) -> str:
        """
        Get a brief summary of the current configuration.
        
        Args:
            current_dataset: Override for CURRENT_DATASET if provided
            current_model: Override for CURRENT_MODEL if provided
            data_config: Override for DATA_CONFIG if provided
        
        Returns:
            String summary of key configuration parameters
        """ 
        dataset_name = CURRENT_DATASET()['data_prefix'] 
        model_name = current_model 
        
        windows = list(DATA_CONFIG()['window_map'].keys())
        
        summary = f"""
Configuration Summary:
- Model: {model_name}
- Task: {TASK_CONFIG()['name']}
- Dataset: {dataset_name}
- Windows: {windows}
- Folds: {TRAINING_CONFIG()['n_folds']}
- Epochs: {TRAINING_CONFIG()['num_epochs']}
- Batch Size: {TRAINING_CONFIG()['batch_size']}
- Learning Rate: {TRAINING_CONFIG()['learning_rate']}
- Device: {DEVICE_CONFIG['device']}
- Scheduler: {TRAINING_CONFIG()['lr_scheduler']['type']}
- Loss: {TRAINING_CONFIG()['loss_function']['type']}
- Threshold Optimization: {TRAINING_CONFIG()['threshold_optimization']['enabled']}
"""
        return summary.strip() 