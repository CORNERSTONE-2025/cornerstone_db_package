import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MaskAwareLSTM(nn.Module):
    """
    Mask-aware LSTM baseline with the same interface as the Transformer:
        forward(X, mask) -> logits

    Args
    ----
    input_size: int               # number of input features per timestep
    hidden_size: int = 256
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.1          # applied to LSTM (between stacked layers) and head
    pooling: str = "mask_mean"    # "mask_mean" | "last"
    use_mask_concatenation: bool = False  # if True, concat mask as an extra feature
    head_hidden: int | None = None        # optional MLP head width; if None, go linear
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
        pooling: str = "mask_mean",
        use_mask_concatenation: bool = False,
        head_hidden: int | None = None,
    ):
        super().__init__()

        assert pooling in {"mask_mean", "last"}

        self.use_mask_concatenation = use_mask_concatenation
        lstm_input_size = input_size + (1 if use_mask_concatenation else 0)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch only applies if num_layers > 1
        )
        
        # Use default PyTorch initialization for now
        self.pooling = pooling
        self.dropout = nn.Dropout(p=dropout)

        feat_out = hidden_size * (2 if bidirectional else 1)

        if head_hidden is None:
            self.head = nn.Linear(feat_out, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_out, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(head_hidden, 1),
            )
    

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        X:    [B, T, F]
        mask: [B, T] binary {0,1} (1=valid timestep). Required.

        Returns:
            logits: [B] (float), suitable for BCEWithLogitsLoss
        """
        B, T, F = X.shape
        if mask is None:
            # Fails fast—this baseline relies on mask for packing and pooling
            raise ValueError("MaskAwareLSTM requires a time mask of shape [B, T].")

        # Replace NaNs and extreme values to avoid numerical instabilities (same as Transformer)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        X = torch.clamp(X, min=-1e3, max=1e3)

        # Handle 3D mask [B, T, F] - convert to time mask [B, T] by checking if any features are valid
        if mask.dim() == 3:
            # Convert feature mask to time mask: valid timestep if any feature is valid
            mask = (mask.sum(dim=-1) > 0).float()  # [B, T]

        # Optional mask concatenation as an input feature for ablation
        if self.use_mask_concatenation:
            mask_expanded = mask.unsqueeze(-1).to(X.dtype)  # [B, T, 1]
            X = torch.cat([X, mask_expanded], dim=-1)  # [B, T, F+1]

        # Compute lengths from mask; clamp to avoid zeros for packing
        lengths = mask.sum(dim=1).long()
        lengths_clamped = torch.clamp(lengths, min=1)

        # Sort by length (required by pack_padded_sequence with enforce_sorted=True)
        lengths_sorted, idx_sort = torch.sort(lengths_clamped, descending=True)
        X_sorted = X.index_select(0, idx_sort)

        # Pack (lengths must be on CPU for pack_padded_sequence)
        packed = pack_padded_sequence(
            X_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )

        # Ensure LSTM parameters are flattened for efficiency
        self.lstm.flatten_parameters()
        packed_out, _ = self.lstm(packed)

        # Pad back to [B, T, H]
        out_seq, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        # Unsort to original order
        idx_unsort = torch.argsort(idx_sort)
        out_seq = out_seq.index_select(0, idx_unsort)  # [B, T, H']

        # Pooling
        if self.pooling == "mask_mean":
            # masked mean over time
            mask_f = mask.unsqueeze(-1).to(out_seq.dtype)  # [B, T, 1]
            denom = torch.clamp(mask_f.sum(dim=1), min=1.0)  # [B, 1]
            pooled = (out_seq * mask_f).sum(dim=1) / denom   # [B, H']
        else:  # "last"
            # take last valid timestep per sample
            # build indices [B, 1, H'] by gathering along time
            last_idx = torch.clamp(lengths - 1, min=0)  # [B]
            # Gather: convert [B] -> [B,1,1] and expand to hidden dim
            gather_idx = last_idx.view(B, 1, 1).expand(-1, 1, out_seq.size(-1))
            pooled = out_seq.gather(dim=1, index=gather_idx).squeeze(1)  # [B, H']

        pooled = self.dropout(pooled) 
        logits = self.head(pooled).squeeze(-1)  # [B] 
        return logits
