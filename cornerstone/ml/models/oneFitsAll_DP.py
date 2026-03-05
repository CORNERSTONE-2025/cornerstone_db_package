from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from .embed_DP import DataEmbedding, DataEmbedding_wo_time


class gpt4ts(nn.Module):
    
    def __init__(self, patch_size = 8 , stride =8, d_model = 768 , dropout = 0.2 , mask_concat = True ): 
        super(gpt4ts, self).__init__()
        self.pred_len = 0  
        self.patch_size =  patch_size
        self.stride = stride
        self.gpt_layers = 6 
        self.num_classes = 1
        self.d_model = d_model
        self.dropout = dropout
        self.mask_concat = mask_concat
        
        self.seq_len = None
        self.feat_dim = None 
        self.patch_num = None
        
        # Patch padding layer (independent)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # Lazy placeholders
        self.enc_embedding = None
        self.ln_proj = None
        self.out_layer = None

        # GPT2 backbone
        self.gpt2 = GPT2Model.from_pretrained(
            'gpt2', 
            output_attentions=True, 
            output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        # Freeze most GPT2 parameters (same as your original)
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout)
        
        
        
    def _lazy_init(self, x, mask  ):
        """Initialize data-dependent layers on first forward."""
        B, L, M = x.shape
        self.seq_len = L
        if self.mask_concat and mask is not None:
            self.feat_dim = M  *2
        else :
            self.feat_dim = M
        

        # Compute patch number dynamically
        self.patch_num = (L - self.patch_size) // self.stride + 2  # +1 after padding

        # Embedding
        self.enc_embedding = DataEmbedding(
            self.feat_dim * self.patch_size,
            self.d_model,
            self.dropout
        ).to(x.device)

        # Normalization + classifier
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num).to(x.device)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, 1).to(x.device)

    def forward(self, x, mask , x_mark_enc=None):
        if self.enc_embedding is None:
            # Default to binary classification if not specified
            self._lazy_init(x, mask)
        if self.mask_concat and mask is not None :     
            x = torch.cat([x, mask], dim=-1).float()
        x_augmented = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x_augmented = torch.clamp(x_augmented, min=-1e3, max=1e3)
        # Compute mean and std over time and batch dimensions
        mean = x_augmented.mean(dim=(0, 1), keepdim=True)
        std = x_augmented.std(dim=(0, 1), keepdim=True)

        # Avoid division by zero
        x_augmented = (x_augmented - mean) / (std + 1e-8)
        
        B, L, M = x.shape
        input_x = rearrange(x_augmented, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        outputs = self.enc_embedding(input_x, None)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return  outputs.squeeze(-1)  # Shape [B], compatible with BCEWithLogitsLoss