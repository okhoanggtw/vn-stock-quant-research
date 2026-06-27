"""
data_loader.py
Module load du lieu gia co phieu Viet Nam tu vnstock (dung API moi vnstock.api).
"""

# pyrefly: ignore [missing-import]
from vnstock.api.quote import Quote
import pandas as pd
import os


def get_stock_history(symbol: str, start: str, end: str, source: str = "VCI") -> pd.DataFrame:
    """
    Lay du lieu OHLCV lich su cho 1 ma co phieu.

    Args:
        symbol: ma co phieu, vd 'FPT'
        start: ngay bat dau 'YYYY-MM-DD'
        end: ngay ket thuc 'YYYY-MM-DD'
        source: nguon du lieu (mac dinh VCI)

    Returns:
        DataFrame voi cac cot: time, open, high, low, close, volume
    """
    q = Quote(symbol=symbol, source=source)
    df = q.history(start=start, end=end, interval="1D")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def save_raw_data(df: pd.DataFrame, symbol: str, output_dir: str = "data/raw") -> str:
    """Luu data raw ra file CSV, tra ve duong dan file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{symbol}_raw.csv")
    df.to_csv(filepath, index=False)
    return filepath