"""
02_features.py
Ngay 2: Tinh technical indicators va tao target cho model.

Chay: python scripts/02_features.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.features import add_technical_indicators, create_target

SYMBOL = "FPT"

if __name__ == "__main__":
    raw_path = f"data/raw/{SYMBOL}_raw.csv"
    print(f"Dang doc: {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=["time"])

    print(f"So dong truoc khi them feature: {df.shape[0]}")

    df = add_technical_indicators(df)
    df = create_target(df, horizon=1)

    print("\n5 dong sau khi them feature:")
    print(df.tail(5)[["time", "close", "MA_10", "MA_20", "RSI_14", "MACD", "target"]])

    so_dong_truoc = df.shape[0]
    df = df.dropna().reset_index(drop=True)
    print(f"\nSo dong sau khi bo NaN: {df.shape[0]} (da bo {so_dong_truoc - df.shape[0]} dong)")

    print(f"\nTi le target: \n{df['target'].value_counts(normalize=True)}")

    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/{SYMBOL}_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDa luu: {out_path}")