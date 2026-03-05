# models/transformer_with_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """
    Extension of PyTorch's TransformerEncoderLayer that saves attention weights during the forward pass.
    
    This allows later inspection or visualization of what the model was attending to at each timestep.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None  # to store attention weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Perform multi-head self-attention and capture attention weights
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,              # tell PyTorch we want the weights
            average_attn_weights=False      # return per-head weights
        )
        self.attn_weights = attn_weights  # Save for later retrieval

        # Standard transformer operations: residual connection, dropout, layer norm, feedforward
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MaskAwareTransformerClassifier(nn.Module):
    """
    Transformer-based classifier for multivariate time series with missing values.

    It augments input features with masks, applies attention, pools valid timesteps, 
    and predicts the probability of a solar flare.
    """
    def __init__(self, input_dim=23, model_dim=128, num_heads=5, num_layers=3, dropout_prob=0.2, temporal_encoding="sinusoidal", use_mask_concatenation=True):
        """
        Args:
            input_dim: Number of input features per timestep.
            model_dim: Embedding size used internally by the Transformer.
            num_heads: Number of attention heads in each Transformer layer.
            num_layers: Number of TransformerEncoderLayers stacked.
            dropout_prob: Dropout probability used throughout the network.
            temporal_encoding: Type of temporal encoding ("none", "sinusoidal", "time_delta").
            use_mask_concatenation: Whether to concatenate mask with input features.
        """
        super().__init__()

        # Store masking configuration
        self.use_mask_concatenation = use_mask_concatenation
        
        # Project input features (and optionally mask) into Transformer dimension
        if use_mask_concatenation:
            self.input_proj = nn.Linear(input_dim * 2, model_dim)  # features + mask
        else:
            self.input_proj = nn.Linear(input_dim, model_dim)  # features only
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

        # Stack multiple Transformer layers with self-attention
        self.attn_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=128,     # size of the feedforward sub-network inside each block
                dropout=dropout_prob,
                batch_first=True,        # (batch, seq, feature) format
                activation="gelu"        # smoother activation than ReLU
            ) for _ in range(num_layers)  
        ])

        # Layer normalization after Transformer blocks
        self.norm = nn.LayerNorm(model_dim)

        # # Classification head: fully connected layers
        # self.fc = nn.Sequential(
        #     nn.Linear(model_dim, 64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(64, 1)  # binary classification output (logit)
        # )
        # Enhanced classification head
        self.fc = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask, return_attention=False, delta_t=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            mask: Mask tensor indicating valid (1) or missing (0) features, same shape as x.
            return_attention: Whether to return attention weights along with the output.
        
        Returns:
            Output logits of shape (batch_size,) or (batch_size, 1),
            optionally attention weights if return_attention=True.
        """
        # ---------------------------------
        # Step 1: Prepare Inputs
        # ---------------------------------

        # Optionally concatenate features and masks (so the model knows where missing data was)
        if self.use_mask_concatenation:
            x_augmented = torch.cat([x, mask], dim=-1).float()
        else:
            x_augmented = x.float()

        # Replace NaNs and extreme values to avoid numerical instabilities
        x_augmented = torch.nan_to_num(x_augmented, nan=0.0, posinf=1e3, neginf=-1e3)
        x_augmented = torch.clamp(x_augmented, min=-1e3, max=1e3)

        # Linear projection to model dimension
        x_proj = self.input_proj(x_augmented)

        # Apply temporal encoding
        x_proj = self.pos_enc(x_proj, mask=mask, delta_t=delta_t)

        # Create attention mask: True where timestep is invalid (completely missing)
        attention_mask = (mask.sum(dim=-1) == 0).bool()

        # ---------------------------------
        # Step 2: Transformer Encoding
        # ---------------------------------
        for layer in self.attn_layers:
            x_proj = layer(x_proj, src_key_padding_mask=attention_mask)

        # Normalize the final embeddings
        x_encoded = self.norm(x_proj)

        # ---------------------------------
        # Step 3: Pooling across Timesteps
        # ---------------------------------

        # Mask out invalid timesteps before pooling
        valid_timesteps = (mask.sum(dim=-1) > 0).float()
        x_masked = x_encoded * valid_timesteps.unsqueeze(-1)  # (batch_size, seq_len, model_dim)

        # Sum across time, then divide by number of valid timesteps (mean pooling)
        pooled = x_masked.sum(dim=1)
        lengths = valid_timesteps.sum(dim=1).unsqueeze(-1)
        x_pooled = pooled / (lengths + 1e-6)  # avoid division by zero

        # ---------------------------------
        # Step 4: Classification
        # ---------------------------------
        out = self.fc(x_pooled)

        if return_attention:
            # Collect all attention weights from all layers
            all_attn = [layer.attn_weights for layer in self.attn_layers if layer.attn_weights is not None]
            return out.squeeze(-1), all_attn

        return out.squeeze(-1)
