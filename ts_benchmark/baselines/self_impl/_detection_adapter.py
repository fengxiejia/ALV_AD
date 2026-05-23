import math
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, win_size: int, stride: int = 1):
        if values.ndim != 2:
            raise ValueError("WindowDataset expects a 2D array.")
        self.values = values.astype(np.float32)
        self.win_size = int(win_size)
        self.stride = max(1, int(stride))
        self.starts = list(range(0, max(0, len(values) - self.win_size + 1), self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        return torch.from_numpy(self.values[start : start + self.win_size])


class DetectionAdapter:
    default_hyper_params = {
        "batch_size": 128,
        "seq_len": 100,
        "num_epochs": 5,
        "lr": 1e-3,
        "patience": 3,
        "train_stride": 1,
        "anomaly_ratio": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
        "latent_dim": 64,
        "hidden_dim": 128,
        "d_model": 128,
        "n_heads": 4,
        "dropout": 0.1,
        "topk": 8,
        "n_memory": 64,
    }

    def __init__(self, **kwargs):
        params = dict(self.default_hyper_params)
        params.update(kwargs)
        self.config = params
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_state = None
        self.best_loss = math.inf

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def _make_model(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        recon = self.model(batch)
        if isinstance(recon, tuple):
            recon = recon[0]
        return F.mse_loss(recon, batch)

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        recon = self.model(batch)
        if isinstance(recon, tuple):
            recon = recon[0]
        return (recon - batch).pow(2).mean(dim=-1)

    def _loader(self, values: np.ndarray, shuffle: bool, stride: int = 1) -> DataLoader:
        dataset = WindowDataset(values, self.config["seq_len"], stride=stride)
        if len(dataset) == 0:
            raise ValueError("Series shorter than seq_len.")
        return DataLoader(
            dataset,
            batch_size=int(self.config["batch_size"]),
            shuffle=shuffle,
            num_workers=0,
        )

    def detect_fit(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        del train_label
        values = train_data.values.astype(np.float32)
        self.scaler.fit(values)
        values = self.scaler.transform(values).astype(np.float32)
        n_features = values.shape[1]
        self.model = self._make_model(n_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["lr"]))
        train_loader = self._loader(values, shuffle=True, stride=int(self.config["train_stride"]))

        patience = int(self.config["patience"])
        stale = 0
        for epoch in range(int(self.config["num_epochs"])):
            self.model.train()
            losses = []
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = self._loss(batch, epoch + 1)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(losses)) if losses else math.inf
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def detect_score(self, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        values = self.scaler.transform(test_data.values.astype(np.float32)).astype(np.float32)
        loader = self._loader(values, shuffle=False, stride=1)
        self.model.eval()
        window_scores = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                score = self._score_batch(batch).detach().cpu().numpy()
                window_scores.append(score)
        window_scores = np.concatenate(window_scores, axis=0)
        point_scores = np.zeros(len(values), dtype=np.float32)
        counts = np.zeros(len(values), dtype=np.float32)
        win = int(self.config["seq_len"])
        for start, scores in enumerate(window_scores):
            point_scores[start : start + win] += scores[:win]
            counts[start : start + win] += 1.0
        point_scores = point_scores / np.maximum(counts, 1.0)
        return point_scores, point_scores

    def detect_label(self, test_data: pd.DataFrame):
        scores, raw_scores = self.detect_score(test_data)
        ratios = self.config["anomaly_ratio"]
        if not isinstance(ratios, list):
            ratios = [ratios]
        preds = {}
        for ratio in ratios:
            threshold = np.percentile(scores, 100.0 - float(ratio))
            preds[ratio] = (scores > threshold).astype(int)
        return preds, raw_scores
