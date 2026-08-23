"""Probe classes for activation-based preference prediction."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from wandb_utils import log_metrics


class MLPProbe(nn.Module):
    """Simple MLP probe for activation classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PreferenceProbe:
    """Wrapper for training and using preference probes."""

    def __init__(self, probe_type: str = "logistic", input_dim: int = None):
        self.probe_type = probe_type
        self.input_dim = input_dim

        if probe_type == "logistic":
            self.probe = None  # Will be created during training
        elif probe_type == "mlp":
            self.probe = MLPProbe(input_dim)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

    def train(self, X: np.ndarray, y: np.ndarray, device: str = "cuda"):
        """Train the probe on activation-label pairs."""
        if self.probe_type == "logistic":
            self.probe = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.probe.fit(X, y)
            preds = self.probe.predict(X)
            probs = self.probe.predict_proba(X)[:, 1]
            acc = accuracy_score(y, preds)

            log_metrics({
                "probe/train_accuracy": acc,
                "probe/train_samples": len(X),
                "probe/input_dim": X.shape[1],
            })

            print(f"Probe training accuracy: {acc:.4f}")
            return acc
        else:
            # MLP training
            self.probe = self.probe.to(device)
            X_t = torch.FloatTensor(X).to(device)
            y_t = torch.FloatTensor(y).to(device)

            optimizer = torch.optim.AdamW(self.probe.parameters(), lr=1e-3)
            criterion = nn.BCEWithLogitsLoss()

            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            self.probe.train()
            for epoch in range(10):
                total_loss = 0
                n_batches = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    logits = self.probe(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                avg_loss = total_loss / n_batches
                log_metrics({"probe/mlp_train_loss": avg_loss}, step=epoch)

            self.probe.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(self.probe(X_t)) > 0.5).cpu().numpy()
            acc = accuracy_score(y, preds)

            log_metrics({
                "probe/train_accuracy": acc,
                "probe/train_samples": len(X),
                "probe/input_dim": X.shape[1],
            })

            print(f"Probe training accuracy: {acc:.4f}")
            return acc

    def predict_proba(self, X: np.ndarray, device: str = "cuda") -> np.ndarray:
        """Get probability scores for samples."""
        if self.probe_type == "logistic":
            return self.probe.predict_proba(X)[:, 1]
        else:
            self.probe.eval()
            self.probe.to(device)
            X_t = torch.FloatTensor(X).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(self.probe(X_t)).cpu().numpy()
            return probs

    def predict(self, X: np.ndarray, device: str = "cuda") -> np.ndarray:
        """Get binary predictions."""
        probs = self.predict_proba(X, device)
        return (probs > 0.5).astype(int)
