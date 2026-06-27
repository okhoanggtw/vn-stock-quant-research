"""
features.py
Module tinh technical indicators tu du lieu OHLCV.
Dung thu vien 'ta' (Technical Analysis Library).
"""

import pandas as pd
# pyrefly: ignore [missing-import]
from ta.momentum import RSIIndicator
# pyrefly: ignore [missing-import]
from ta.trend import MACD, SMAIndicator
# pyrefly: ignore [missing-import]
from ta.volatility import BollingerBands


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Them cac technical indicators vao DataFrame OHLCV.

    Args:
        df: DataFrame co cot 'close' (gia dong cua)

    Returns:
        DataFrame voi cac cot moi: MA_10, MA_20, RSI_14,
        MACD, MACD_signal, BB_high, BB_low
    """
    df = df.copy()

    # --- Moving Average (MA) ---
    df["MA_10"] = SMAIndicator(close=df["close"], window=10).sma_indicator()
    df["MA_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()

    # --- RSI (Relative Strength Index) ---
    df["RSI_14"] = RSIIndicator(close=df["close"], window=14).rsi()

    # --- MACD ---
    macd = MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # --- Bollinger Bands ---
    bb = BollingerBands(close=df["close"], window=20)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    return df


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Tao target (nhan) de model hoc: ngay mai gia tang (1) hay giam (0).
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["target"] = (future_close > df["close"]).astype(int)
    return df