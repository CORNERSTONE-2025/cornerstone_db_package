# models/transformer.py
import torch
import torch.nn as nn
import math

class IdentityPositionalEncoding(nn.Module):
    """No-op positional encoding - makes model order-invariant across time axis."""
    def forward(self, x, mask=None, delta_t=None):
        # x: [B, T, d_model]
        return x

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x, mask=None, delta_t=None):
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class TimeDeltaEncoding(nn.Module):
    """Simple Δt -> d_model MLP; if delta_t is None, defaults to unit steps."""
    def __init__(self, d_model, log_scale=True):
        super().__init__()
        self.log_scale = log_scale
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, mask=None, delta_t=None):
        B, T, _ = x.shape
        if delta_t is None:
            # Fallback: assume unit stride
            dt = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        else:
            # expect delta_t as [B, T] or [B, T, 1]
            if delta_t.dim() == 2:
                dt = delta_t.unsqueeze(-1).to(dtype=x.dtype)
            else:
                dt = delta_t.to(dtype=x.dtype)
        if self.log_scale:
            dt = torch.log1p(dt)
        pe = self.mlp(dt)  # [B, T, d_model]
        return x + pe

class MaskAwareTransformerClassifier(nn.Module):
    def __init__(self, input_dim=30, model_dim=64, num_heads=4, num_layers=2, dropout_prob=0.3, temporal_encoding="sinusoidal"):
        super().__init__()
        self.input_proj = nn.Linear(input_dim * 2, model_dim)
        self.temporal_encoding = temporal_encoding.lower()
        
        # Choose temporal encoding
        if self.temporal_encoding == "none":
            self.pos_enc = IdentityPositionalEncoding()
        elif self.temporal_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(model_dim)
        elif self.temporal_encoding == "time_delta":
            self.pos_enc = TimeDeltaEncoding(model_dim)
        else:
            raise ValueError(f"Unknown temporal_encoding: {self.temporal_encoding}")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout_prob,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1)
        )


    def forward(self, x, mask, delta_t=None):
        x_augmented = torch.cat([x, mask], dim=-1).float()
        x_augmented = torch.nan_to_num(x_augmented, nan=0.0, posinf=1e3, neginf=-1e3)
        x_augmented = torch.clamp(x_augmented, min=-1e3, max=1e3)
        x_proj = self.input_proj(x_augmented)
        
        # Apply temporal encoding
        x_proj = self.pos_enc(x_proj, mask=mask, delta_t=delta_t)
        
        attention_mask = (mask.sum(dim=-1) == 0).bool()
        x_encoded = self.encoder(x_proj, src_key_padding_mask=attention_mask)
        valid_timesteps = (mask.sum(dim=-1) > 0).float()
        x_masked = x_encoded * valid_timesteps.unsqueeze(-1)
        pooled = x_masked.sum(dim=1)
        lengths = valid_timesteps.sum(dim=1).unsqueeze(-1)
        x_pooled = pooled / (lengths + 1e-6)
        out = self.fc(x_pooled)
        return out.squeeze(-1)
