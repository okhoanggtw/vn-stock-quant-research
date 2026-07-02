"""
models.py
Module dung de dinh nghia, train va evaluate cac model Machine Learning va Deep Learning (LSTM PyTorch).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =====================================================================
# 1. Classical Machine Learning Models
# =====================================================================

def get_ml_model(model_type: str = "rf", random_state: int = 42):
    """
    Tra ve instance cua model Machine Learning.
    """
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=random_state, n_jobs=-1
        )
    elif model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, random_state=random_state
        )
    else:
        raise ValueError(f"Model type {model_type} khong duoc ho tro.")


def evaluate_ml_ts_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Danh gia model ML bang TimeSeriesSplit de tranh look-ahead bias (ro ri du lieu tuong lai).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        
        # Calculate performance
        metrics.append({
            "fold": fold + 1,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc": roc_auc_score(y_test, probs)
        })
        
    return pd.DataFrame(metrics)


# =====================================================================
# 2. Deep Learning Models (PyTorch LSTM)
# =====================================================================

class TimeSeriesDataset(Dataset):
    """
    Dataset loader cho PyTorch dung cho du lieu dang chuoi thoi gian.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len, num_features)
        # y: (N,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_lstm_sequences(X_scaled: np.ndarray, y: np.ndarray, seq_len: int = 10):
    """
    Bien doi du lieu 2D thanh dang 3D (samples, sequence_length, features) de train LSTM.
    """
    Xs, ys = [], []
    for i in range(len(X_scaled) - seq_len):
        Xs.append(X_scaled[i : (i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


class LSTMClassifier(nn.Module):
    """
    Kien truc model LSTM cho bai toan phan loai xu huong gia co phieu (Up/Down).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 1, dropout: float = 0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Layer de dua ra xac suat
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass qua LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Lay output cua buoc thoi gian cuoi cung (last sequence step)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


def train_lstm_epoch(model, dataloader, criterion, optimizer, device):
    """
    Chay 1 epoch training cho model LSTM.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward
        preds = model(X_batch).squeeze()
        
        # Loss
        loss = criterion(preds, y_batch)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        
    return total_loss / len(dataloader.dataset)


def predict_lstm(model, X_seq: np.ndarray, device) -> np.ndarray:
    """
    Du doan probability tang gia bang model LSTM.
    """
    model.eval()
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).squeeze().cpu().numpy()
    
    # Neu tra ve gia tri don le (batch size = 1)
    if preds.ndim == 0:
        return np.array([preds.item()])
    return preds
