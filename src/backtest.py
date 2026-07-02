"""
backtest.py
Module thuc hien backtest chien luoc giao dich va tinh toan cac chi so do luong hieu qua (Sharpe, Sortino, Drawdown...).
"""

import numpy as np
import pandas as pd


def run_vectorized_backtest(df: pd.DataFrame, signal_col: str, tc: float = 0.0015) -> pd.DataFrame:
    """
    Backtest vectorized cho chien luoc Long-Only hoac Long-Short.
    
    Args:
        df: DataFrame goc co chua cot 'close' va cot tin hieu signal.
        signal_col: ten cot chua tin hieu (vd: 'signal' gia tri 0 hoac 1)
        tc: transaction cost (phi giao dich + thue, mac dinh 0.15% moi chieu mua/ban)
        
    Returns:
        DataFrame bo sung cac cot returns va cumulative returns.
    """
    df = df.copy()
    
    # Tinh toan return cua thi truong (mua va nam giu)
    df["market_return"] = df["close"].pct_change()
    
    # Dich tin hieu di 1 ngay de tranh look-ahead bias
    # Tin hieu phat sinh tai ngay t chi co the thuc thi vao ngay t+1
    df["signal_shifted"] = df[signal_col].shift(1)
    
    # Tinh return chien luoc truoc chi phi
    df["strategy_return_raw"] = df["signal_shifted"] * df["market_return"]
    
    # Tinh chi phi giao dich (khi thay doi tin hieu tu 0 -> 1 hoac 1 -> 0)
    # Khoang cach tuyet doi cua signal cho biet quy mo giao dich
    df["trade_size"] = (df["signal_shifted"] - df["signal_shifted"].shift(1)).abs()
    df["transaction_cost"] = df["trade_size"] * tc
    
    # Return thuc te sau phi
    df["strategy_return"] = df["strategy_return_raw"] - df["transaction_cost"].fillna(0)
    
    # Tinh cumulative returns (quy doi ve gia tri tich luy tu 1 dong ban dau)
    df["market_cum"] = (1 + df["market_return"].fillna(0)).cumprod() - 1
    df["strategy_cum"] = (1 + df["strategy_return"].fillna(0)).cumprod() - 1
    
    return df


def calculate_metrics(returns: pd.Series) -> dict:
    """
    Tinh toan cac quantitative performance metrics tu chuoi returns hang ngay.
    """
    returns = returns.fillna(0)
    n_days = len(returns)
    if n_days == 0:
        return {}
        
    # 1. Total Return
    total_return = (1 + returns).prod() - 1
    
    # 2. Annualized Return (Gia su 252 ngay giao dich/nam)
    ann_factor = 252 / n_days
    ann_return = (1 + total_return) ** ann_factor - 1 if total_return > -1 else -1.0
    
    # 3. Sharpe Ratio (risk-free rate = 0)
    daily_std = returns.std()
    sharpe = (np.sqrt(252) * returns.mean() / daily_std) if daily_std > 0 else 0.0
    
    # 4. Sortino Ratio (chi xet rui ro giam gia - downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (np.sqrt(252) * returns.mean() / downside_std) if downside_std > 0 else 0.0
    
    # 5. Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = drawdowns.min()
    
    # 6. Win Rate (chi xet cac ngay co vi the va co bien dong)
    active_days = returns[returns != 0]
    win_rate = (active_days > 0).sum() / len(active_days) if len(active_days) > 0 else 0.0
    
    # 7. Profit Factor
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    profit_factor = (gains / abs(losses)) if losses != 0 else (gains if gains > 0 else 1.0)
    
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor
    }
