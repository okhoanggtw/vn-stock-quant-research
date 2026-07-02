"""
04_train_dl.py
Ngay 8: Xay dung va train model Deep Learning (LSTM) bang PyTorch.
Ap dung chuan hoa StandardScaler va chia sequence 3D phu hop cho mang Recurrent Neural Network.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    LSTMClassifier,
    TimeSeriesDataset,
    create_lstm_sequences,
    train_lstm_epoch,
    predict_lstm
)

SYMBOL = "FPT"
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
HIDDEN_DIM = 32
NUM_LAYERS = 2


def main():
    # 1. Load data
    processed_path = f"data/processed/{SYMBOL}_features.csv"
    if not os.path.exists(processed_path):
        print(f"Loi: Khong tim thay file {processed_path}. Vui long chay scripts/02_features.py.")
        return

    print(f"--- 1. LOADING DATA FOR LSTM TRAINING ---")
    df = pd.read_csv(processed_path, parse_dates=["time"])
    
    # 2. Extract features & target
    exclude_cols = ["time", "target", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df["target"].values

    # 3. Sequential Split
    split_idx = int(len(df) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

    print(f"Train raw shape: {X_train_raw.shape}, Test raw shape: {X_test_raw.shape}")

    # 4. Feature Standardization (StandardScaler fit tren Train, transform cho Test)
    # De tranh data leakage, khong fit scaler tren toan bo dataset!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 5. Create Sequences (samples, sequence_length, features)
    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train_raw, seq_len=SEQ_LEN)
    X_test_seq, y_test_seq = create_lstm_sequences(X_test_scaled, y_test_raw, seq_len=SEQ_LEN)

    print(f"Train sequence shape: {X_train_seq.shape}, target: {y_train_seq.shape}")
    print(f"Test sequence shape: {X_test_seq.shape}, target: {y_test_seq.shape}")

    # 6. Create PyTorch DataLoader
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 7. Khoi tao model, loss function va optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice dang su dung: {device}")
    
    model = LSTMClassifier(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.2
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 8. Training loop
    print("\n--- 2. TRAINING PYTORCH LSTM MODEL ---")
    for epoch in range(1, EPOCHS + 1):
        loss = train_lstm_epoch(model, train_loader, criterion, optimizer, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{EPOCHS} - Loss: {loss:.6f}")

    # 9. Evaluate model tren tap Test
    print("\n--- 3. EVALUATING LSTM ON TEST SET ---")
    test_probs = predict_lstm(model, X_test_seq, device)
    test_preds = (test_probs > 0.5).astype(int)

    test_acc = accuracy_score(y_test_seq, test_preds)
    test_auc = roc_auc_score(y_test_seq, test_probs)

    print(f"LSTM Test Accuracy: {test_acc:.4f}")
    print(f"LSTM Test AUC-ROC: {test_auc:.4f}")

    # 10. Save Model, Scaler va list features dung de inference sau nay
    os.makedirs("models", exist_ok=True)
    
    # Save PyTorch Model weights
    torch_model_path = f"models/{SYMBOL}_lstm.pt"
    torch.save(model.state_dict(), torch_model_path)
    
    # Save Scaler va Metadata
    scaler_path = f"models/{SYMBOL}_lstm_scaler.pkl"
    joblib.dump({
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }, scaler_path)
    
    print(f"\nDa luu LSTM model tai: {torch_model_path}")
    print(f"Da luu Scaler va Metadata tai: {scaler_path}")


if __name__ == "__main__":
    main()
