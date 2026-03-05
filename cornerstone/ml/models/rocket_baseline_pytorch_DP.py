import torch
import torch.nn as nn
import torch.nn.functional as F

class RocketTorch(nn.Module):
    """
    Minimal PyTorch-native ROCKET layer.
    Applies random 1D conv kernels + global max + PPV (proportion of positive values) pooling.
    """
    def __init__(self, input_channels, num_kernels=10000, kernel_size=9):
        super().__init__()
        self.num_kernels = num_kernels
        self.input_channels = input_channels
        self.kernel_size = kernel_size

        # Random 1D convolution kernels
        self.convs = nn.ModuleList([
            nn.Conv1d(input_channels, 1, kernel_size, bias=False, padding=0)
            for _ in range(num_kernels)
        ])
        # Initialize kernels uniformly like ROCKET
        for conv in self.convs:
            conv.weight.data.uniform_(-1, 1)

    def forward(self, x):
        # x: [B, C, T]
        feats = []
        for conv in self.convs:
            out = conv(x)  # [B, 1, T']
            feats.append(out.max(dim=-1).values)  # global max pooling
            feats.append((out > 0).float().mean(dim=-1))  # PPV pooling
        return torch.cat(feats, dim=1)  # [B, 2*num_kernels]

class MaskAwareROCKET(nn.Module):
    """
    Mask-aware ROCKET baseline using PyTorch-native ROCKET.
    forward(X, mask) -> logits
    """
    def __init__(
        self,
        input_size: int,
        num_kernels: int = 10000,
        pooling: str = "mask_mean",
        dropout: float = 0.1,
        head_hidden: int | None = None,
        kernel_size: int = 9
    ):
        super().__init__()

        self.pooling = pooling
        self.dropout = nn.Dropout(p=dropout)

        # PyTorch-native ROCKET
        self.rocket = RocketTorch(input_channels=input_size, num_kernels=num_kernels, kernel_size=kernel_size)
        feat_out = 2 * num_kernels  # MAX + PPV per kernel

        # usare classiier di scikit (logistic)
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
        B, T, F = X.shape
        if mask is None:
            raise ValueError("MaskAwareROCKET requires a time mask of shape [B, T].")

        # Replace NaNs / clamp extremes
        X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        X = torch.clamp(X, min=-1e3, max=1e3)

        # Convert feature mask [B, T, F] -> time mask [B, T]
        if mask.dim() == 3:
            mask = (mask.sum(dim=-1) > 0).float()

        lengths = mask.sum(dim=1).long()
        max_len = max(lengths.max().item(), self.rocket.kernel_size)
        X_trimmed = torch.zeros((B, max_len, F), device=X.device, dtype=X.dtype)
        for i in range(B):
            L = lengths[i].item()
            X_trimmed[i, :L, :] = X[i, :L, :]

        # ROCKET expects [B, C, T]
        X_trimmed = X_trimmed.permute(0, 2, 1)  # [B, F, T]

        # Extract ROCKET features
        feats = self.rocket(X_trimmed)

        pooled = self.dropout(feats)
        logits = self.head(pooled).squeeze(-1)  # [B]
        return logits
