import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[10, 20, 40], bottleneck_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False) \
            if in_channels > 1 else nn.Identity()

        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels if in_channels > 1 else in_channels,
                      out_channels,
                      kernel_size=k,
                      padding=k // 2,
                      bias=False)
            for k in kernel_sizes
        ])

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x) if not isinstance(self.bottleneck, nn.Identity) else x
        conv_outs = [conv(z) for conv in self.convs]

        pool_out = self.conv_pool(self.maxpool(x))
        target_len = conv_outs[0].size(-1)
        if pool_out.size(-1) != target_len:
            diff = target_len - pool_out.size(-1)
            pool_out = F.pad(pool_out, (0, diff)) if diff > 0 else pool_out[:, :, :target_len]

        out = torch.cat(conv_outs + [pool_out], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in range(depth):
            self.blocks.append(InceptionBlock(
                in_channels if d == 0 else out_channels * 4,
                out_channels
            ))

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out

class MaskAwareInceptionTime(nn.Module):
    def __init__(self, input_size, nb_filters=32, depth=6, dropout=0.1, head_hidden=None):
        super().__init__()
        self.inception = InceptionModule(
            in_channels=input_size,
            out_channels=nb_filters,
            depth=depth
        )

        self.dropout = nn.Dropout(p=dropout)
        feat_dim = nb_filters * 4

        # Single output like ROCKET
        if head_hidden is None:
            self.head = nn.Linear(feat_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(head_hidden, 1),
            )

    def masked_avg_pool(self, x, mask):
        B, C, T = x.shape
        if mask.size(1) != T:
            mask = F.interpolate(mask[:, None, :].float(), size=T, mode="nearest").squeeze(1)
        mask = mask.unsqueeze(1)
        x = x * mask
        summed = x.sum(dim=-1)
        lengths = mask.sum(dim=-1).clamp(min=1)
        return summed / lengths

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, F = X.shape
        if mask is None:
            raise ValueError("MaskAwareInceptionTime requires a time mask of shape [B, T].")

        X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        X = torch.clamp(X, min=-1e3, max=1e3)

        if mask.dim() == 3:
            mask = (mask.sum(dim=-1) > 0).float()

        lengths = mask.sum(dim=1).long()
        max_len = lengths.max().item()
        X_trimmed = torch.zeros((B, max_len, F), device=X.device, dtype=X.dtype)
        mask_trimmed = torch.zeros((B, max_len), device=X.device, dtype=mask.dtype)
        for i in range(B):
            L = lengths[i].item()
            X_trimmed[i, :L, :] = X[i, :L, :]
            mask_trimmed[i, :L] = mask[i, :L]

        X_trimmed = X_trimmed.transpose(1, 2)
        feats = self.inception(X_trimmed)
        pooled = self.masked_avg_pool(feats, mask_trimmed)
        pooled = self.dropout(pooled)
        logits = self.head(pooled).squeeze(-1)  # [B], same as ROCKET
        return logits