"""
01_get_data.py
Ngay 1: Lay du lieu gia co phieu va xem qua data.

Chay: python scripts/01_get_data.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_stock_history, save_raw_data
import matplotlib.pyplot as plt

SYMBOL = "FPT"
START = "2023-01-01"
END = "2025-06-23"

if __name__ == "__main__":
    print(f"Dang lay du lieu cho ma: {SYMBOL} ...")
    df = get_stock_history(SYMBOL, START, END)

    print(f"\nSo dong: {df.shape[0]}, So cot: {df.shape[1]}")
    print("\n10 dong dau:")
    print(df.head(10))

    filepath = save_raw_data(df, SYMBOL)
    print(f"\nDa luu raw data: {filepath}")

    # Ve chart gia dong cua
    plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["close"])
    plt.title(f"Gia dong cua {SYMBOL}")
    plt.xlabel("Ngay")
    plt.ylabel("Gia (VND)")
    os.makedirs("reports", exist_ok=True)
    plt.savefig(f"reports/{SYMBOL}_price_chart.png")
    print(f"Da luu chart: reports/{SYMBOL}_price_chart.png")
