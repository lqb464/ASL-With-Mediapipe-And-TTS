from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml


with open("configs/model.yaml", encoding="utf-8") as f:
    model_cfg = yaml.safe_load(f)["model"]

@dataclass(frozen=True)
class SequenceRNNConfig:
    input_dim: int
    num_classes: int
    model_type: str = model_cfg["type"]
    hidden_dim: int = model_cfg["hidden_dim"]
    num_layers: int = model_cfg["num_layers"]
    dropout: float = model_cfg["dropout"]
    bidirectional: bool = model_cfg["bidirectional"]
    seed: int = model_cfg["seed"]

class SequenceRNNClassifier(nn.Module):
    def __init__(self, config: SequenceRNNConfig):
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        rnn_cls = nn.GRU if config.model_type.lower() == "gru" else nn.LSTM

        self.rnn = rnn_cls(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        fc_input_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.fc = nn.Linear(fc_input_dim, config.num_classes)

    def forward(self, x, lengths=None):
        out, _ = self.rnn(x)

        if lengths is None:
            last_out = out[:, -1, :]
        else:
            lengths = lengths.clamp(min=1)
            batch_idx = torch.arange(out.size(0), device=out.device)
            last_idx = lengths.to(out.device) - 1
            last_out = out[batch_idx, last_idx, :]

        return self.fc(last_out)

    def save(self, path: Path, extra: Dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": {
                "input_dim": self.config.input_dim,
                "num_classes": self.config.num_classes,
                "model_type": self.config.model_type,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "bidirectional": self.config.bidirectional,
                "seed": self.config.seed,
            },
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str | Path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        config = SequenceRNNConfig(**payload["config"])
        model = SequenceRNNClassifier(config)
        model.load_state_dict(payload["state_dict"])
        return model