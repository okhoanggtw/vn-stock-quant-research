"""
features.py
Module tinh technical indicators tu du lieu OHLCV va kiem tra tinh dung (stationarity).
Dung thu vien 'ta' va 'statsmodels'.
"""

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from ta.momentum import RSIIndicator
# pyrefly: ignore [missing-import]
from ta.trend import MACD, SMAIndicator
# pyrefly: ignore [missing-import]
from ta.volatility import BollingerBands, AverageTrueRange
# pyrefly: ignore [missing-import]
from statsmodels.tsa.stattools import adfuller


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Them cac technical indicators va lag features vao DataFrame OHLCV.

    Args:
        df: DataFrame co cac cot 'open', 'high', 'low', 'close', 'volume'

    Returns:
        DataFrame da bo sung cac features
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

    # --- Volatility (Average True Range - ATR) ---
    df["ATR_14"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    # --- Log Returns ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- Lag Features (Dung de lam prediction) ---
    # Lag cua close returns, RSI va MACD giup model hoc tu lich su
    for lag in [1, 2, 3]:
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
        df[f"RSI_lag_{lag}"] = df["RSI_14"].shift(lag)
        df[f"MACD_lag_{lag}"] = df["MACD"].shift(lag)

    return df


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Tao target (nhan) de model hoc: ngay mai gia tang (1) hay giam (0).
    Sử dụng horizon ngày để dự đoán trước.
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["target"] = (future_close > df["close"]).astype(int)
    return df


def check_stationarity(series: pd.Series) -> dict:
    """
    Thuc hien kiem dinh Augmented Dickey-Fuller (ADF) de kiem tra tinh dung.
    Mot chuoi thoi gian stationary giup cac model hoc tot hon, tranh spurious regression.
    """
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return {"is_stationary": False, "p_value": 1.0}
    result = adfuller(series_clean)
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05
    }