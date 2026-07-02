"""
06_portfolio_opt.py
Nice to Have: Portfolio Optimization & Risk Management.
Tai du lieu nhieu ma co phieu (FPT, HPG, VNM, SSI, TCB), dung solver SLSQP trong Scipy
de tinh toan ty trong danh muc toi uu (Max Sharpe, Min Variance).
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_stock_history
from src.portfolio import optimize_portfolio, compute_portfolio_stats

# Universe co phieu VN30 bieu dien cho cac nhom nganh khac nhau:
# Cong nghe (FPT), Thep (HPG), Tieu dung (VNM), Chung khoan (SSI), Ngan hang (TCB)
TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]
START = "2023-01-01"
END = "2025-06-23"
RF = 0.04  # Lai suat phi rui ro nam gia su la 4% (lai suat tiet kiem VND)


def main():
    print(f"--- 1. FETCHING HISTORICAL DATA FOR PORTFOLIO CONSTITUTION ---")
    prices_dict = {}
    
    # Lay du lieu gia close cua tung ticker
    import time
    for symbol in TICKERS:
        try:
            print(f"Dang lay du lieu: {symbol}...")
            df = get_stock_history(symbol, START, END)
            prices_dict[symbol] = df.set_index("time")["close"]
            time.sleep(3.5)  # Tranh chan API (Rate Limit 20 req/min)
        except Exception as e:
            print(f"Loi lay du lieu {symbol}: {e}")
            time.sleep(3.5)
            
    # Gop thanh 1 DataFrame chung
    prices_df = pd.DataFrame(prices_dict).sort_index()
    # Fill missing values bang forward fill (tranh lech ngay giao dich le)
    prices_df = prices_df.ffill().dropna()
    
    print(f"\nDa hop nhat du lieu. So dong: {len(prices_df)} ngay.")
    print("5 dong dau gia dong cua cua cac co phieu:")
    print(prices_df.head(5))

    # 2. Tinh toan daily log returns
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

    # 3. Tinh toan gia tri trung binh (mean) va ma tran hiep phuong sai (covariance) hang ngay
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    print("\nTysuat sinh loi ky vong hang ngay (trung binh):")
    print(mean_returns.round(6))
    print("\nMa tran hiep phuong sai (Covariance Matrix):")
    print(cov_matrix.round(6))

    # 4. Thuc hien toi uu hoa danh muc
    print("\n--- 2. SOLVING PORTFOLIO OPTIMIZATION PROBLEM (SLSQP Solver) ---")
    
    # 4.1 Equal Weight (Danh muc co ty trong deu)
    eq_weights = np.array([1.0 / len(TICKERS)] * len(TICKERS))
    
    # 4.2 Max Sharpe Ratio Portfolio
    max_sharpe_weights = optimize_portfolio(mean_returns, cov_matrix, objective="max_sharpe", rf=RF)
    
    # 4.3 Minimum Variance Portfolio
    min_var_weights = optimize_portfolio(mean_returns, cov_matrix, objective="min_variance", rf=RF)

    # 5. Tinh toan thong so thong ke va so sanh
    eq_ret, eq_vol, eq_sharpe = compute_portfolio_stats(eq_weights, mean_returns, cov_matrix, rf=RF)
    ms_ret, ms_vol, ms_sharpe = compute_portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix, rf=RF)
    mv_ret, mv_vol, mv_sharpe = compute_portfolio_stats(min_var_weights, mean_returns, cov_matrix, rf=RF)

    # 6. In ra bang so sanh
    portfolio_summary = pd.DataFrame({
        "Ticker": TICKERS,
        "Equal Weight (%)": (eq_weights * 100).round(2),
        "Max Sharpe Weights (%)": (max_sharpe_weights * 100).round(2),
        "Min Variance Weights (%)": (min_var_weights * 100).round(2)
    })
    
    print("\n==========================================================================")
    print("                     PORTFOLIO WEIGHT ALLOCATIONS                         ")
    print("==========================================================================")
    print(portfolio_summary.to_string(index=False))
    print("==========================================================================")

    stats_summary = pd.DataFrame({
        "Metric (Annualized)": ["Expected Return", "Volatility (Risk)", "Sharpe Ratio (Rf=4%)"],
        "Equal Weight Portfolio": [f"{eq_ret*100:.2f}%", f"{eq_vol*100:.2f}%", f"{eq_sharpe:.3f}"],
        "Maximum Sharpe Portfolio": [f"{ms_ret*100:.2f}%", f"{ms_vol*100:.2f}%", f"{ms_sharpe:.3f}"],
        "Minimum Variance Portfolio": [f"{mv_ret*100:.2f}%", f"{mv_vol*100:.2f}%", f"{mv_sharpe:.3f}"]
    })
    
    print("\n==========================================================================")
    print("                     PORTFOLIO PERFORMANCE STATISTICS                     ")
    print("==========================================================================")
    print(stats_summary.to_string(index=False))
    print("==========================================================================")

    # 7. Ve bieu do phan bo ty trong (Chi hien thi nhan cho cac ma co ty trong > 1% de tranh chong cheo)
    labels_ms = [TICKERS[i] if w > 0.01 else "" for i, w in enumerate(max_sharpe_weights)]
    labels_mv = [TICKERS[i] if w > 0.01 else "" for i, w in enumerate(min_var_weights)]
    
    def my_autopct(pct):
        return f"{pct:.1f}%" if pct > 1.0 else ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Max Sharpe Weights
    axes[0].pie(max_sharpe_weights, labels=labels_ms, autopct=my_autopct, startangle=90, colors=plt.cm.tab20(np.arange(len(TICKERS)) % 20))
    axes[0].set_title(f"Max Sharpe Allocation (Sharpe: {ms_sharpe:.2f})\nRet: {ms_ret*100:.1f}%, Vol: {ms_vol*100:.1f}%")
    
    # Chart 2: Min Variance Weights
    axes[1].pie(min_var_weights, labels=labels_mv, autopct=my_autopct, startangle=90, colors=plt.cm.tab20(np.arange(len(TICKERS)) % 20))
    axes[1].set_title(f"Min Variance Allocation (Sharpe: {mv_sharpe:.2f})\nRet: {mv_ret*100:.1f}%, Vol: {mv_vol*100:.1f}%")
    
    plt.suptitle("Portfolio Optimization Results (VND Assets)")
    plt.tight_layout()
    
    os.makedirs("reports", exist_ok=True)
    chart_path = "reports/portfolio_allocation_chart.png"
    plt.savefig(chart_path)
    print(f"\nDa ve va luu bieu do phan bo danh muc tai: {chart_path}")


if __name__ == "__main__":
    main()
