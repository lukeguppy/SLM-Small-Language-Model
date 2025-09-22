import torch
import torch.nn as nn
from typing import List, Union

from .custom_attention import CustomTransformerEncoder
from ..model_config import ModelConfig


class SmallTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Validate config parameters
        if not config.validate():
            raise ValueError("Invalid model configuration parameters")

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)  # max sentence length
        self.dropout = nn.Dropout(config.dropout)

        # Use custom transformer with attention weight capture
        self.transformer = CustomTransformerEncoder(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            num_layers=config.n_layers,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
        )
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

        # Store model parameters for attention extraction
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers

    def forward(self, x, mask=None, return_attention=False, return_embeddings=False):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        x = self.dropout(x)
        # No transpose needed with batch_first=True

        # Causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Handle padding mask
        if mask is not None:
            # mask: True = keep, False = pad → invert for transformer
            transformer_mask = ~mask
            out = self.transformer(x, mask=causal_mask, src_key_padding_mask=transformer_mask)
        else:
            out = self.transformer(x, mask=causal_mask)

        # Output is already (batch, seq_len, embed)
        if return_embeddings:
            # Return embeddings (transformer output) instead of logits
            if return_attention:
                attention_weights = self.get_attention_weights()
                return out, attention_weights
            return out

        logits = self.fc_out(out)

        if return_attention:
            # Extract real attention weights from the custom transformer
            attention_weights = self.get_attention_weights()
            return logits, attention_weights

        return logits

    def get_attention_weights(self, layer_idx=None) -> Union[torch.Tensor, List[torch.Tensor], None]:
        """Extract attention weights from the transformer layers."""
        return self.transformer.get_attention_weights(layer_idx)
