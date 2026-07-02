"""
03_train_ml.py
Ngay 3-4: Xay dung, train va evaluate model Machine Learning (Random Forest).
Su dung TimeSeriesSplit de cross-validate va kiem tra tinh dung cua du lieu (ADF test).
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.features import check_stationarity
from src.models import get_ml_model, evaluate_ml_ts_cv

SYMBOL = "FPT"


def main():
    # 1. Load data
    processed_path = f"data/processed/{SYMBOL}_features.csv"
    if not os.path.exists(processed_path):
        print(f"Loi: Khong tim thay file {processed_path}. Vui long chay scripts/02_features.py truoc.")
        return

    print(f"--- 1. LOADING PROCESSED DATA FOR {SYMBOL} ---")
    df = pd.read_csv(processed_path, parse_dates=["time"])
    print(f"Data shape: {df.shape}")

    # 2. Statistical Analysis: ADF test de kiem tra tinh dung (Stationarity)
    print("\n--- 2. RUNNING STATISTICAL STATIONARITY TEST (ADF) ---")
    
    # Kiem tra gia close (thuong la non-stationary)
    close_adf = check_stationarity(df["close"])
    print(f"Close Price ADF p-value: {close_adf['p_value']:.4f} (Is stationary? {close_adf['is_stationary']})")
    
    # Kiem tra log return (thuong la stationary)
    return_adf = check_stationarity(df["log_return"])
    print(f"Log Return ADF p-value: {return_adf['p_value']:.4f} (Is stationary? {return_adf['is_stationary']})")
    
    if return_adf["is_stationary"]:
        print("=> Nhan xet: Gia dong cua khong dung, nhung ti suat sinh loi (Log Return) la chuoi dung. "
              "Day la ly do chung ta su dung log return va cac technical indicators lam dac trung (features) de train model.")
    else:
        print("=> Can than: Log return chua hoan toan dung trong tap du lieu nay.")

    # 3. Chuan bi features va target
    print("\n--- 3. PREPARING FEATURES & TARGET ---")
    # Loai bo cac cot thong tin chung hoac ranh gioi
    exclude_cols = ["time", "target", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df["target"]
    
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    # 4. Sequential Split (Train/Test) de tranh look-ahead leakage
    # Vi day la du lieu chuoi thoi gian, khong dung train_test_split ngau nhien!
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)} ({df['time'].iloc[0].strftime('%Y-%m-%d')} den {df['time'].iloc[split_idx-1].strftime('%Y-%m-%d')})")
    print(f"Test size: {len(X_test)} ({df['time'].iloc[split_idx].strftime('%Y-%m-%d')} den {df['time'].iloc[-1].strftime('%Y-%m-%d')})")

    # 5. Time-Series Cross Validation tren tap Train
    print("\n--- 4. TRAINING WITH TIME-SERIES CROSS VALIDATION ---")
    model = get_ml_model("rf")
    cv_results = evaluate_ml_ts_cv(model, X_train, y_train, n_splits=5)
    print("\nKet qua Cross Validation qua tung Fold:")
    print(cv_results.round(4))
    print(f"\nAverage CV Accuracy: {cv_results['accuracy'].mean():.4f}")
    print(f"Average CV AUC: {cv_results['auc'].mean():.4f}")

    # 6. Train tren toan bo tap Train va evaluate tap Test
    print("\n--- 5. EVALUATING ON TEST SET ---")
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    
    print(f"Test Set Accuracy: {test_acc:.4f}")
    print(f"Test Set AUC-ROC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, zero_division=0))

    # 7. Save model va feature importance plot
    os.makedirs("models", exist_ok=True)
    model_filepath = f"models/{SYMBOL}_rf.pkl"
    joblib.dump(model, model_filepath)
    print(f"\nDa luu model ML tai: {model_filepath}")

    # Plot & Save Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {SYMBOL} Random Forest")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_cols[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    os.makedirs("reports", exist_ok=True)
    chart_path = f"reports/{SYMBOL}_feature_importance.png"
    plt.savefig(chart_path)
    print(f"Da ve va luu bieu do Feature Importance tai: {chart_path}")


if __name__ == "__main__":
    main()
