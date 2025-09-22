import torch
import math
from typing import Optional, List, Union

import torch.nn as nn
import torch.nn.functional as F


class CustomMultiHeadAttention(nn.Module):
    """Custom Multi-Head Attention layer that captures and stores attention weights."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Attention weights storage
        self.attention_weights = None

    def forward(self, query, key, value, mask=None, need_weights=True):
        """Forward pass with attention weight capture."""
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)

        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply mask if provided
        if mask is not None:
            # mask shape: (batch_size, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len_q, seq_len_k)

            # Expand mask to match scores shape
            mask = mask.expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Store attention weights for visualisation
        self.attention_weights = attention_weights

        # Apply dropout
        attention_weights = self.dropout_layer(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(context)

        if need_weights:
            return output, attention_weights
        else:
            return output, None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights."""
        return self.attention_weights


class CustomTransformerEncoderLayer(nn.Module):
    """Custom Transformer Encoder Layer using CustomMultiHeadAttention."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalisation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation function - store as callable
        self.activation_fn = activation

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Forward pass for transformer encoder layer."""
        # Self-attention with residual connection
        # Always capture attention weights for visualisation purposes
        src2, _ = self.self_attn(src, src, src, mask=src_mask, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network with residual connection
        src2 = self.linear2(self.dropout(self.activation_fn(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from the self-attention layer."""
        return self.self_attn.get_attention_weights()


class CustomTransformerEncoder(nn.Module):
    """Custom Transformer Encoder that can capture attention weights from all layers."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()
        # Create layers with provided parameters
        self.layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """Forward pass through all encoder layers."""
        output = src
        for layer in self.layers:
            output = layer(output, mask, src_key_padding_mask)
        return output

    def get_attention_weights(self, layer_idx: Optional[int] = None) -> Union[torch.Tensor, List[torch.Tensor], None]:
        """Get attention weights from specified layer or all layers."""
        if layer_idx is not None:
            if 0 <= layer_idx < self.num_layers:
                # Type: ignore, this is a CustomTransformerEncoderLayer at runtime
                return self.layers[layer_idx].get_attention_weights()  # type: ignore
            else:
                return None
        else:
            attention_weights: List[torch.Tensor] = []
            for i in range(self.num_layers):
                # Type: ignore, this is a CustomTransformerEncoderLayer at runtime
                weights = self.layers[i].get_attention_weights()  # type: ignore
                if weights is not None:
                    attention_weights.append(weights)
            return attention_weights if attention_weights else None
