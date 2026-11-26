"""
lstm_model.py — CrowdGuard AI
===============================
Configurable LSTM model for crowd anomaly / risk classification.

Architecture mirrors GRU model for fair comparison:
    Input sequence (batch, seq_len, input_dim)
    → LSTM layer(s)
    → Dropout
    → FC hidden layer (optional)
    → Output logits (batch, output_dim)

Design notes:
  - Returns raw logits — use CrossEntropyLoss.
  - Bidirectional False by default.
  - LSTM has cell state (c_t) in addition to hidden state (h_t).
  - Same initialisation strategy as GRU for fair comparison.
"""

import torch
import torch.nn as nn
from typing import Optional


class CrowdGuardLSTM(nn.Module):
    """
    LSTM-based temporal classifier for crowd risk estimation.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        LSTM hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    output_dim : int
        Number of output classes.
    dropout : float
        Dropout probability.
    fc_hidden_dim : int
        Intermediate FC layer size; 0 to skip.
    bidirectional : bool
        Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.3,
        fc_hidden_dim: int = 32,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.config = {
            "model_type": "lstm",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "dropout": dropout,
            "fc_hidden_dim": fc_hidden_dim,
            "bidirectional": bidirectional,
        }

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        lstm_out_dim = hidden_dim * self.n_directions
        if fc_hidden_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(lstm_out_dim, fc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_hidden_dim, output_dim),
            )
        else:
            self.fc = nn.Linear(lstm_out_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier/orthogonal init + forget-gate bias = 1."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # LSTM forget gate (gate index 1 of 4)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        logits : (batch, output_dim)
        """
        out, (h_n, c_n) = self.lstm(x)
        # Use the last time step output
        last = out[:, -1, :]          # (batch, hidden*n_dir)
        last = self.dropout(last)
        logits = self.fc(last)        # (batch, output_dim)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lstm_from_config(cfg: dict, input_dim: Optional[int] = None) -> CrowdGuardLSTM:
    """Build LSTM from a parsed YAML config dict."""
    arch = cfg.get("architecture", {})
    idim = input_dim or arch.get("input_dim")
    if idim is None:
        raise ValueError("input_dim must be provided either via config or argument.")
    return CrowdGuardLSTM(
        input_dim=idim,
        hidden_dim=arch.get("hidden_dim", 64),
        num_layers=arch.get("num_layers", 2),
        output_dim=arch.get("output_dim", 2),
        dropout=arch.get("dropout", 0.3),
        fc_hidden_dim=arch.get("fc_hidden_dim", 32),
        bidirectional=arch.get("bidirectional", False),
    )
