import sys
import os
import importlib
import torch
       
LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Time_Series_Library'))
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class MaskAwareTimesSeriesLibrary(torch.nn.Module):
    # Time_Series_Library configs wrapper
    class ModelConfig:
        def __init__(self, config_dict):
            for k, v in config_dict.items():
                setattr(self, k, v)

    MODEL_EXTRA_ARGS = {
        "Reformer": {"bucket_size": 4, "n_hashes": 4},
        "Pyraformer": {"window_size": [4,4], "inner_size": 5},
        "LightTS": {"chunk_size": 8},
        "FEDFormer": {"version": "fourier", "mode_select": "random", "modes": 32},
        "DLinear": {"individual": True},
    }

    def __init__(self, **kwargs):
        super().__init__()
        model_name = kwargs.get("model_name", None)
        if model_name is None:
            raise ValueError("'model_name' is required as input parameter of the configuration.")
        
        module_path = f"Time_Series_Library.models.{model_name}"
        old_layers = sys.modules.get('layers', None)
        old_utils = sys.modules.get('utils', None)
        sys.modules['layers'] = importlib.import_module('Time_Series_Library.layers')
        sys.modules['utils'] = importlib.import_module('Time_Series_Library.utils')

        try:
            model_module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            raise ValueError(f"""'{model_name}' not found in Time_Series_Library.models
                            Accepted models:
                            Autoformer, DLinear, ETSformer, FEDformer, Informer, LightTS, 
                            Nonstationary_Transformer, Pyraformer, Reformer, TimesNet""")
        finally:
            # Ripristino i moduli originali
            if old_layers is not None:
                sys.modules['layers'] = old_layers
            else:
                del sys.modules['layers']
            if old_utils is not None:
                sys.modules['utils'] = old_utils
            else:
                del sys.modules['utils']

        self.model_params = kwargs.copy()
        if self.model_params.get("use_mask_concatenation", True):
            if "enc_in" in self.model_params:
                self.model_params["enc_in"] += 1
            if "dec_in" in self.model_params:
                self.model_params["dec_in"] += 1

        configs = self.ModelConfig(self.model_params)
        extra_args = {}
        for arg, default_val in self.MODEL_EXTRA_ARGS.get(model_name, {}).items():
            extra_args[arg] = self.model_params.get(arg, default_val)
        self.model = model_module.Model(configs, **extra_args)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if x_mark_enc is None:
            raise ValueError("MaskAwareTimesSeriesLibrary requires a time mask of shape [B, T].")

        # Replace NaNs and extreme values to avoid numerical instabilities (same as Transformer)
        x_enc = torch.nan_to_num(x_enc, nan=0.0, posinf=1e3, neginf=-1e3)
        x_enc = torch.clamp(x_enc, min=-1e3, max=1e3)

        # Handle 3D mask [B, T, F] - convert to time mask [B, T] by checking if any features are valid
        if x_mark_enc.dim() == 3:
            x_mark_enc = (x_mark_enc.sum(dim=-1) > 0).float()  # [B, T]

        # Optional mask concatenation as an input feature for ablation
        use_mask_concat = self.model_params.get("use_mask_concatenation", False)
        if use_mask_concat:
            mask_feature = x_mark_enc.unsqueeze(-1).to(x_enc.dtype)  # [B, T, 1]
            x_enc = torch.cat([x_enc, mask_feature], dim=-1)  # [B, T, F+1]

        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask).squeeze(-1)