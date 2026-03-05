import torch
from typing import Optional, Dict, Any, Type
from ..models.transformer_with_attention import MaskAwareTransformerClassifier
from ..models.lstm_baseline import MaskAwareLSTM
from ..models.time_series_library_with_mask import MaskAwareTimesSeriesLibrary
from ..models.prova import MaskAwareUSRLWrapper
from .config import MODEL_CONFIGS, DEVICE_CONFIG, get_model_config
from ..models.rocket_DP import MaskAwareROCKET #DP
from ..models.inceptionTime_baseline_DP import MaskAwareInceptionTime #DP
from ..models.oneFitsAll_DP import gpt4ts #DP 
from ..models.XGBoost_DP import MaskAwareXGBoost #DP
from ..models.miniRocket_DP import MaskAwareMINIROCKET 


class ModelManager:
    """Manages model creation, loading, and saving."""
    
    # Map of model class names to their implementations
    MODEL_CLASSES = {
        "TRANSFORMER": MaskAwareTransformerClassifier,
        "TRANSFORMER_ATTENTION": MaskAwareTransformerClassifier,
        "LSTM": MaskAwareLSTM,
        "TimesNet": MaskAwareTimesSeriesLibrary,
        "Reformer": MaskAwareTimesSeriesLibrary,
        "Pyraformer": MaskAwareTimesSeriesLibrary,
        "Nonstationary_Transformer": MaskAwareTimesSeriesLibrary,
        "LightTS": MaskAwareTimesSeriesLibrary,
        "Informer": MaskAwareTimesSeriesLibrary,
        "FEDformer": MaskAwareTimesSeriesLibrary,
        "ETSformer": MaskAwareTimesSeriesLibrary,
        "DLinear": MaskAwareTimesSeriesLibrary,
        "Autoformer": MaskAwareTimesSeriesLibrary,
        "USRL": MaskAwareUSRLWrapper,
        "ROCKET": MaskAwareROCKET, #DP
        "INCEPTIONTIME":  MaskAwareInceptionTime,#DP
        "ONEFITSALL" : gpt4ts, # DP
        "XGBOOST" : MaskAwareXGBoost,
        "MINIROCKET" : MaskAwareMINIROCKET
    }
    
    def __init__(self, current_model, device: Optional[str] = None):
        self.device = device or DEVICE_CONFIG["device"]
        # Get current model from the loaded config
        import sys
        #current_model = getattr(sys.modules.get('run_config', None), 'CURRENT_MODEL', 'TRANSFORMER_ATTENTION')
        self.model_class = self.MODEL_CLASSES[current_model]
        
    def create_model(self, current_model, **kwargs) -> torch.nn.Module:
        """Creates a new model instance with configured hyperparameters.
        
        Args:
            **kwargs: Override default model parameters from config
        """
        # Get config from loaded module or fall back to global config
        import sys
        if 'run_config' in sys.modules:
            config = sys.modules['run_config'].MODEL_CONFIGS.get(current_model, {})
        else:
            config = get_model_config(current_model) 
            print("Final model parameters:", config)
        # Use provided kwargs or fall back to config
        
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        print(clean_kwargs)
        model_params = {**config, **clean_kwargs}  # kwargs override config values
        print("Final model parameters:", model_params)
        
        # Handle different parameter mappings for different model types
        import sys
        #current_model = getattr(sys.modules.get('run_config', None), 'CURRENT_MODEL', 'TRANSFORMER_ATTENTION')
        if current_model == "LSTM":
            lstm_params = self._map_to_lstm_params(model_params)
            model = self.model_class(**lstm_params)
        else:
            # Instantiate ONCE
            print("Final model parameters:", model_params) 
            model = self.model_class(**model_params)

        # Decide device behavior based on attribute existence
        if hasattr(model, "to"):
            print(f"✅ Loaded PyTorch model on {self.device}")
            return model.to(self.device)
        else:
            print(f"✅ Loaded sklearn model (runs on CPU)")
            return model
                
    def _map_to_lstm_params(self, transformer_params: Dict[str, Any]) -> Dict[str, Any]:
        """Maps transformer parameters to LSTM parameters."""
        lstm_params = {
            "input_size": transformer_params.get("input_dim", 22),
            "hidden_size": transformer_params.get("model_dim", 256),
            "num_layers": transformer_params.get("num_layers", 1),
            "bidirectional": transformer_params.get("bidirectional", True),
            "dropout": transformer_params.get("dropout_prob", 0.1),
            "pooling": transformer_params.get("pooling", "mask_mean"),
            "use_mask_concatenation": transformer_params.get("use_mask_concatenation", False),
            "head_hidden": transformer_params.get("head_hidden", None),
        }
        return lstm_params
    
    def load_model(self, model_path: str, **kwargs) -> torch.nn.Module:
        """Loads a model from a file.
        
        Args:
            model_path: Path to the model file
            **kwargs: Additional parameters for model creation
        """
        model = self.create_model(**kwargs)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def save_model(self, model: torch.nn.Module, path: str) -> None:
        """Saves a model to a file."""
        torch.save(model.state_dict(), path)
    
    def get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Gets information about a model."""
        return {
            "type": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": next(model.parameters()).device
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], device: Optional[str] = None) -> 'ModelManager':
        """Creates a ModelManager from a configuration dictionary."""
        model_class_name = config.get("model_class", "MaskAwareTransformerClassifier")
        model_class = cls.MODEL_CLASSES.get(model_class_name, MaskAwareTransformerClassifier)
        return cls(device=device, model_class=model_class) 