"""
05_backtest.py
Ngay 7: Backtest va so sanh hieu qua giua cac chien luoc (Random Forest, LSTM, va Buy-and-Hold).
Ap dung phi giao dich 0.15% moi giao dich de mo phong thuc te, tinh toan Sharpe, Sortino, Drawdown.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import LSTMClassifier, predict_lstm
from src.backtest import run_vectorized_backtest, calculate_metrics

SYMBOL = "FPT"


def main():
    # 1. Load data
    processed_path = f"data/processed/{SYMBOL}_features.csv"
    if not os.path.exists(processed_path):
        print(f"Loi: Khong tim thay file {processed_path}. Vui long chay cac buoc truoc.")
        return

    df = pd.read_csv(processed_path, parse_dates=["time"])
    split_idx = int(len(df) * 0.8)

    # 2. Load models
    rf_model_path = f"models/{SYMBOL}_rf.pkl"
    lstm_scaler_path = f"models/{SYMBOL}_lstm_scaler.pkl"
    lstm_weights_path = f"models/{SYMBOL}_lstm.pt"

    if not os.path.exists(rf_model_path) or not os.path.exists(lstm_weights_path):
        print("Loi: Khong tim thay file model da train. Vui long chay scripts/03_train_ml.py va scripts/04_train_dl.py.")
        return

    print("--- 1. LOADING TRAINED MODELS ---")
    # Load Random Forest
    rf_model = joblib.load(rf_model_path)
    print("Random Forest model loaded.")

    # Load LSTM Scaler & Metadata
    lstm_meta = joblib.load(lstm_scaler_path)
    scaler = lstm_meta["scaler"]
    feature_cols = lstm_meta["feature_cols"]
    seq_len = lstm_meta["seq_len"]
    hidden_dim = lstm_meta["hidden_dim"]
    num_layers = lstm_meta["num_layers"]
    print("LSTM Scaler & Metadata loaded.")

    # Load LSTM Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMClassifier(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=device))
    lstm_model.eval()
    print("LSTM PyTorch model weights loaded.")

    # 3. Chuan bi du lieu kiem thu (Test set) dong nhat cho ca 2 model
    # Tap Test bat dau tu index: split_idx
    # Tuy nhien LSTM can 'seq_len' ngay lich su de tao sequence dau tien,
    # vi vay tap test thuc te bat dau tu split_idx + seq_len
    start_test_idx = split_idx + seq_len
    test_df = df.iloc[start_test_idx:].copy().reset_index(drop=True)
    
    print(f"\nSo dong test thuc te dung de so sanh: {len(test_df)} ngay "
          f"({test_df['time'].iloc[0].strftime('%Y-%m-%d')} den {test_df['time'].iloc[-1].strftime('%Y-%m-%d')})")

    # --- Predict bang Random Forest ---
    X_test_ml = df[feature_cols].iloc[start_test_idx:]
    rf_probs = rf_model.predict_proba(X_test_ml)[:, 1]
    test_df["rf_signal"] = (rf_probs > 0.5).astype(int)

    # --- Predict bang LSTM ---
    # Chuan hoa features bang scaler da luu cua LSTM
    X_all_raw = df[feature_cols].values
    X_all_scaled = scaler.transform(X_all_raw)
    
    # Lay cua so truoc do de tao dung chuoi du lieu sequence cho LSTM
    # X_test_lstm_raw chua du 10 ngay lich su cho moi diem trong tap test_df
    X_test_seqs = []
    for i in range(start_test_idx, len(df)):
        X_test_seqs.append(X_all_scaled[i - seq_len : i])
    X_test_seqs = np.array(X_test_seqs) # Shape: (len(test_df), seq_len, num_features)

    lstm_probs = predict_lstm(lstm_model, X_test_seqs, device)
    test_df["lstm_signal"] = (lstm_probs > 0.5).astype(int)

    # 4. Chay Backtest (Vectorized backtest, transaction cost = 0.15%)
    print("\n--- 2. RUNNING BACKTESTS (Transaction Cost = 0.15%) ---")
    tc_rate = 0.0015  # 0.15% phi giao dich
    
    # Chay backtest cho Random Forest
    test_df = run_vectorized_backtest(test_df, "rf_signal", tc=tc_rate)
    # Doi ten cot ket qua RF
    test_df = test_df.rename(columns={
        "strategy_return": "rf_return",
        "strategy_cum": "rf_cum"
    })
    
    # Chay backtest cho LSTM
    test_df = run_vectorized_backtest(test_df, "lstm_signal", tc=tc_rate)
    # Doi ten cot ket qua LSTM
    test_df = test_df.rename(columns={
        "strategy_return": "lstm_return",
        "strategy_cum": "lstm_cum"
    })

    # 5. Tinh toan metrics cho tung chien luoc
    market_metrics = calculate_metrics(test_df["market_return"])
    rf_metrics = calculate_metrics(test_df["rf_return"])
    lstm_metrics = calculate_metrics(test_df["lstm_return"])

    # 6. Hien thi ket qua so sanh
    metrics_summary = pd.DataFrame({
        "Metric": ["Total Return", "Annualized Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Profit Factor"],
        "Buy & Hold": [
            f"{market_metrics['total_return']*100:.2f}%",
            f"{market_metrics['ann_return']*100:.2f}%",
            f"{market_metrics['sharpe']:.3f}",
            f"{market_metrics['sortino']:.3f}",
            f"{market_metrics['max_drawdown']*100:.2f}%",
            f"{market_metrics['win_rate']*100:.2f}%",
            f"{market_metrics['profit_factor']:.2f}"
        ],
        "Random Forest Strategy": [
            f"{rf_metrics['total_return']*100:.2f}%",
            f"{rf_metrics['ann_return']*100:.2f}%",
            f"{rf_metrics['sharpe']:.3f}",
            f"{rf_metrics['sortino']:.3f}",
            f"{rf_metrics['max_drawdown']*100:.2f}%",
            f"{rf_metrics['win_rate']*100:.2f}%",
            f"{rf_metrics['profit_factor']:.2f}"
        ],
        "LSTM Strategy": [
            f"{lstm_metrics['total_return']*100:.2f}%",
            f"{lstm_metrics['ann_return']*100:.2f}%",
            f"{lstm_metrics['sharpe']:.3f}",
            f"{lstm_metrics['sortino']:.3f}",
            f"{lstm_metrics['max_drawdown']*100:.2f}%",
            f"{lstm_metrics['win_rate']*100:.2f}%",
            f"{lstm_metrics['profit_factor']:.2f}"
        ]
    })
    
    print("\n==========================================================================")
    print("                      STRATEGY PERFORMANCE COMPARISON                     ")
    print("==========================================================================")
    print(metrics_summary.to_string(index=False))
    print("==========================================================================")

    # 7. Ve bieu do hieu suat tich luy (Equity Curve)
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["time"], test_df["market_cum"] * 100, label="Buy & Hold (Market)", color="gray", linestyle="--")
    plt.plot(test_df["time"], test_df["rf_cum"] * 100, label="Random Forest Strategy", color="blue")
    plt.plot(test_df["time"], test_df["lstm_cum"] * 100, label="LSTM PyTorch Strategy", color="orange")
    
    plt.title(f"Cumulative Returns Comparison - {SYMBOL} ({tc_rate*100}% Transaction Cost)")
    plt.xlabel("Ngay")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("reports", exist_ok=True)
    chart_path = f"reports/{SYMBOL}_backtest_equity.png"
    plt.savefig(chart_path)
    print(f"\nDa ve va luu bieu do Equity Curve tai: {chart_path}")


if __name__ == "__main__":
    main()
