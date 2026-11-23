"""Model definition and inference wrapper for the tuned Module 8 network."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CreditRiskNet(nn.Module):
    """Same MLP as Module 8: Linear+BatchNorm+ReLU blocks, dropout on all
    hidden blocks except the last, single-logit output."""

    def __init__(self, n_features, hidden=(64, 32, 16, 8), dropout=0.3):
        super().__init__()
        layers, prev = [], n_features
        for i, width in enumerate(hidden):
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.ReLU()]
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CreditRiskPredictor:
    def __init__(self, artifact_dir="artifacts"):
        artifact_dir = Path(artifact_dir)
        with open(artifact_dir / "scaler.json") as f:
            cfg = json.load(f)
        self.features = cfg["features"]
        self.mean = np.asarray(cfg["mean"], dtype=np.float32)
        self.scale = np.asarray(cfg["scale"], dtype=np.float32)
        self.threshold = float(cfg["threshold"])
        self.net = CreditRiskNet(len(self.features), tuple(cfg["hidden"]), cfg["dropout"])
        state = torch.load(artifact_dir / "model.pt", map_location="cpu")
        self.net.load_state_dict(state)
        self.net.eval()

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if X.shape[1] != len(self.features):
            raise ValueError(f"Expected {len(self.features)} features, got {X.shape[1]}")
        Xs = (X - self.mean) / self.scale
        with torch.no_grad():
            logits = self.net(torch.from_numpy(Xs)).numpy().ravel()
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_one(self, record: dict):
        row = [float(record[name]) for name in self.features]
        p = float(self.predict_proba([row])[0])
        return p, int(p >= self.threshold)
