# chemberta_model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path
from typing import Union

_CHECKPOINT = "seyonec/ChemBERTa-zinc-base-v1"   # must match training run


class ChemBERTaRegressor(nn.Module):
    """Backbone + small MLP head for affinity regression."""
    def __init__(self, checkpoint: str = _CHECKPOINT):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).pooler_output          # CLS token
        return self.regressor(x).squeeze(-1)


def load_model(
    weight_path: Union[str, Path] = "chemberta_affinity_torch.pt",
    device: Union[str, torch.device, None] = None,
    checkpoint: str = _CHECKPOINT,
) -> ChemBERTaRegressor:
    """
    Reconstruct the network and load weights.

    Example
    -------
    >>> model = load_model("chemberta_affinity_torch.pt")
    >>> model(input_ids, attention_mask)        # ready for inference
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ChemBERTaRegressor(checkpoint).to(dev)
    state = torch.load(weight_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model
