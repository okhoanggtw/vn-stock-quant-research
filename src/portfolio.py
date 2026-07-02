"""
portfolio.py
Module toi uu hoa danh muc dau tu (Portfolio Optimization) su dung Scipy de giai bai toan Maximize Sharpe hoac Min Variance.
"""

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from scipy.optimize import minimize


def compute_portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, rf: float = 0.0) -> tuple:
    """
    Tinh toan ty suat sinh loi ky vong, do lech chuan (volatility) va Sharpe Ratio cua danh muc.
    """
    # Annualized portfolio return
    port_return = np.sum(mean_returns * weights) * 252
    
    # Annualized portfolio volatility
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    # Sharpe Ratio
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0
    
    return port_return, port_vol, sharpe


def _neg_sharpe_objective(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, rf: float) -> float:
    """Objective function de toi da hoa Sharpe ratio (minimize negative Sharpe)."""
    port_return, port_vol, _ = compute_portfolio_stats(weights, mean_returns, cov_matrix, rf)
    return - (port_return - rf) / port_vol if port_vol > 0 else 0.0


def _variance_objective(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, rf: float) -> float:
    """Objective function de toi thieu hoa bien dong (variance)."""
    _, port_vol, _ = compute_portfolio_stats(weights, mean_returns, cov_matrix, rf)
    return port_vol ** 2


def optimize_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame, objective: str = "max_sharpe", rf: float = 0.0) -> np.ndarray:
    """
    Giai bai toan toi uu hoa danh muc bang phuong phap lap SLSQP (Sequential Least Squares Programming).
    
    Args:
        mean_returns: loi nhuan trung binh hang ngay cua cac ma.
        cov_matrix: ma tran hiep phuong sai hang ngay giua cac ma.
        objective: muc tieu toi uu 'max_sharpe' hoac 'min_variance'.
        rf: risk-free rate hang nam (mac dinh 0.0).
        
    Returns:
        numpy.ndarray chua ty trong (weights) cua tung ma co phieu trong danh muc.
    """
    num_assets = len(mean_returns)
    
    # Ty trong khoi tao bang nhau (Equal Weight)
    init_weights = np.array([1.0 / num_assets] * num_assets)
    
    # Ranh gioi cho moi ma co phieu (Long-only: tu 0% den 100%)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Rang buoc: Tong ty trong phai bang 1.0 (100%)
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    })
    
    # Lua chon objective function
    if objective == "max_sharpe":
        obj_fun = _neg_sharpe_objective
    elif objective == "min_variance":
        obj_fun = _variance_objective
    else:
        raise ValueError(f"Objective '{objective}' khong hop le.")
        
    # Giai bai toan toi uu
    res = minimize(
        fun=obj_fun,
        x0=init_weights,
        args=(mean_returns, cov_matrix, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    
    if not res.success:
        print(f"Warning: Toi uu hoa khong thanh cong! Ly do: {res.message}")
        
    # Bo tron va lam sach ty trong (tranh bi so am vo cung nho do tinh toan float)
    weights = np.clip(res.x, 0.0, 1.0)
    weights = weights / np.sum(weights)  # Re-normalize
    
    return weights
