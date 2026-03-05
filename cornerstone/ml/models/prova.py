import sys
import os
import importlib.machinery
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Retrocompatibilità Python 3.12+ per vecchi moduli ---
if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def _find_module(self, fullname):
        spec = self.find_spec(fullname)
        if spec and spec.loader:
            return spec.loader
        return None
    importlib.machinery.FileFinder.find_module = _find_module

# --- Aggiungi la repo USRL all'inizio del sys.path per risolvere conflitti "utils" ---
repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "UnsupervisedScalableRepresentationLearningTimeSeries")
)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

# --- Stampa per verifica ---
'''
print("=== sys.path verifica ===")
for i, p in enumerate(sys.path[:5]):  # mostriamo solo i primi 5 per chiarezza
    print(f"{i}: {p}")
print("=========================")
'''

# --- Importa utils.py della libreria come modulo indipendente ---
usrl_utils_spec = importlib.util.spec_from_file_location(
    "usrl_utils", os.path.join(repo_root, "utils.py")
)
usrl_utils = importlib.util.module_from_spec(usrl_utils_spec)
usrl_utils_spec.loader.exec_module(usrl_utils)
#print("DEBUG: usrl_utils path =", usrl_utils.__file__)

'''
# --- Inseriscilo temporaneamente nel namespace 'losses' / 'scikit_wrappers' ---
sys.modules["UnsupervisedScalableRepresentationLearningTimeSeries.utils"] = usrl_utils

module_path = "UnsupervisedScalableRepresentationLearningTimeSeries.scikit_wrappers"
model_module = importlib.import_module(module_path)
EncoderClass = getattr(model_module, "CausalCNNEncoderClassifier")
'''

# --- Ora importa scikit_wrappers ma "temporaneamente" reindirizzando utils ---
old_utils = sys.modules.get("utils", None)  # salva la tua versione

# PUNTIAMO TEMPORANEAMENTE il nome 'utils' alla versione della libreria
sys.modules["utils"] = usrl_utils

try:
    module_path = "UnsupervisedScalableRepresentationLearningTimeSeries.scikit_wrappers"
    model_module = importlib.import_module(module_path)
    EncoderClass = getattr(model_module, "CausalCNNEncoderClassifier")
finally:
    # Ripristina la tua utils (fondamentale!)
    if old_utils is not None:
        sys.modules["utils"] = old_utils
    else:
        del sys.modules["utils"]


class MaskAwareUSRLWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.use_mask_concat = kwargs.get("use_mask_concatenation", False)
        in_channels = kwargs.get("in_channels", 1)
        #out_channels = kwargs.get("out_channels", 1)
        if self.use_mask_concat:
            in_channels += 1
            #out_channels += 1
            kwargs["in_channels"] = in_channels
            #kwargs["out_channels"] = out_channels

        # Parametri del modello (valori di default presi da scikit_wrappers)
        self.model = EncoderClass(
            compared_length=kwargs.get("compared_length", 50),
            nb_random_samples=kwargs.get("nb_random_samples", 10),
            negative_penalty=kwargs.get("negative_penalty", 1),
            batch_size=kwargs.get("batch_size", 1),
            nb_steps=kwargs.get("nb_steps", 2000),
            lr=kwargs.get("lr", 0.001),
            penalty=kwargs.get("penalty", 1),
            early_stopping=kwargs.get("early_stopping", None),
            channels=kwargs.get("channels", 10),
            depth=kwargs.get("depth", 1),
            reduced_size=kwargs.get("reduced_size", 10),
            out_channels=kwargs.get("out_channels", 10),
            kernel_size=kwargs.get("kernel_size", 4),
            in_channels=in_channels,
            cuda=kwargs.get("cuda", False),
            gpu=kwargs.get("gpu", 0),
        )
        self.model.encoder = self.model.encoder.float()

        # Layer di classificazione binaria
        #self.classifier = nn.Linear(kwargs.get("out_channels", 10), 1)
        out_dim = kwargs.get("out_channels", 10)
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x, mask):
        #print(f"[DEBUG] BEFORE mask handling - x.shape: {x.shape}, mask.shape: {mask.shape if mask is not None else None}")
        if mask is None:
            raise ValueError("MaskAwareUSRLWrapper requires a time mask of shape [B, T].")

        # Replace NaNs and extreme values to avoid numerical instabilities (same as Transformer)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = torch.clamp(x, min=-1e3, max=1e3)

        x = (x - x.mean(dim=(0,1), keepdim=True)) / (x.std(dim=(0,1), keepdim=True) + 1e-6)

        # Handle 3D mask [B, T, F] - convert to time mask [B, T] by checking if any features are valid
        if mask.dim() == 3:
            mask = (mask.sum(dim=-1) > 0).float()  # [B, T]
        
        # Gestione della mask come feature aggiuntiva (opzionale)
        if self.use_mask_concat:
            mask = mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
            x = torch.cat([x, mask], dim=-1)  # [B, T, F+1]
        #print(f"[DEBUG] AFTER mask handling - x.shape: {x.shape}, mask.shape: {mask.shape if mask is not None else None}")
        '''
        # x deve essere numpy (come nel repo originale)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().astype(np.float32)

        x = np.transpose(x, (0, 2, 1))  # [B, F, T]
        features = self.model.encode(x)  # [B, out_channels]
        #print(f"[DEBUG] features mean={features.mean():.6f}, std={features.std():.6f}")
        features = torch.tensor(features, dtype=torch.float32)  # numpy -> tensor
        logits = self.classifier(features)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs#, logits
        '''
        # Assicurati che x abbia la shape [B, C, L] per il CausalCNNEncoder
        #if x.ndim == 3 and x.shape[2] != self.model.in_channels:
        x = x.permute(0, 2, 1)  # [B, F, T] → [B, C, L]
        #print(f"[DEBUG] BEFORE encoder - x.shape: {x.shape}, expected in_channels: {self.model.in_channels}")
        # Usa direttamente l'encoder (torch Module) invece di .encode()
        features = self.model.encoder(x)  # [B, out_channels]
        features = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + 1e-6)
        logits = self.classifier(features)
        #print(f"[DEBUG] logits: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
        probs = torch.sigmoid(logits).squeeze(-1)
        #print(f"[DEBUG] AFTER forward - probs.shape: {probs.shape}")
        return probs  # , logits
