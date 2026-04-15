"""
model.py

Phase 1: MLP baseline classifier on flow features.
Phase 2 stub: Transformer encoder for byte-level packet sequences (PacketFormer).

The MLP is what you train now. The PacketFormer class is scaffolded
and will be filled out in Phase 2 once MI300X access is available.
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# Phase 1: MLP Baseline
# ─────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Feed-forward classifier on pre-extracted CICIDS2017 flow features.
    Establishes the F1 benchmark that PacketFormer needs to beat.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Phase 2: PacketFormer (Transformer Encoder)
# ─────────────────────────────────────────────

class ByteEmbedding(nn.Module):
    """
    Embeds raw byte values (0–255) + positional encoding.
    Input: (batch, seq_len) integer byte tokens
    Output: (batch, seq_len, d_model) embeddings
    """

    def __init__(self, d_model: int = 512, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(256 + 2, d_model)  # 256 bytes + PAD + MASK
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(x) + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PacketFormer(nn.Module):
    """
    Byte-level transformer encoder pre-trained on raw network traffic.

    Phase 2 architecture:
    - Pre-train: masked token prediction on unlabeled captures
    - Fine-tune: flow-level attack classification

    Targets AMD MI300X (192GB) for training at scale.
    Model size: 125M–350M parameters.

    NOTE: This class is scaffolded. Full pre-training loop
    and ROCm-specific optimizations are Phase 2 work.
    """

    PAD_TOKEN = 256
    MASK_TOKEN = 257

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        num_classes: int = None,  # None = pre-training mode
    ):
        super().__init__()

        self.embedding = ByteEmbedding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pre-training head: predict masked byte tokens
        self.mlm_head = nn.Linear(d_model, 256)

        # Fine-tuning head: classify attack type from [CLS] token
        self.classifier = nn.Linear(d_model, num_classes) if num_classes else None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None,
        mode: str = "pretrain",
    ) -> torch.Tensor:
        """
        x: (batch, seq_len) byte token ids
        padding_mask: (batch, seq_len) bool, True = padded position
        mode: 'pretrain' | 'finetune'
        """
        embeddings = self.embedding(x)
        hidden = self.encoder(embeddings, src_key_padding_mask=padding_mask)

        if mode == "pretrain":
            # Return per-token logits for masked token prediction
            return self.mlm_head(hidden)

        elif mode == "finetune":
            if self.classifier is None:
                raise ValueError("Set num_classes to use finetune mode.")
            # Use mean pooling over non-padded positions as sequence representation
            if padding_mask is not None:
                mask = (~padding_mask).unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(dim=1)
            return self.classifier(pooled)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
