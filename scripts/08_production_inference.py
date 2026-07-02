"""
08_production_inference.py
Live Production Inference & Portfolio Rebalancing Pipeline.
Script nay chay hang ngay (cron job) de load model da train, lay du lieu cuoi ngay moi nhat,
dua ra tin hieu trade, toi uu hoa danh muc va xuat lệnh giao dich (Order execution).
Co ho tro gui thong bao qua Telegram Bot.
"""

import sys
import os
import json
import joblib
import datetime
import urllib.request
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_stock_history
from src.features import add_technical_indicators
from src.models import LSTMClassifier, predict_lstm
from src.portfolio import optimize_portfolio, compute_portfolio_stats

# Thong tin cau hinh production
TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]
PRIMARY_SYMBOL = "FPT"
PORTFOLIO_VALUE_VND = 100_000_000  # Danh muc gia su tri gia 100 trieu VND
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def check_data_quality(df: pd.DataFrame, symbol: str) -> bool:
    """Buoc 2: Data Quality check truoc khi dua vao model."""
    if df is None or len(df) < 30:
        print(f"[ERROR] Du lieu cho {symbol} qua ngan hoac bi thieu.")
        return False
    # Kiem tra gia dong cua co NaN khong
    if df["close"].isnull().any():
        print(f"[WARNING] Phat hien gia tri NaN trong cot close cua {symbol}, dang tien hanh forward fill.")
        df["close"] = df["close"].ffill()
    return True


def send_telegram_notification(message: str):
    """Buoc 5: Gui thong bao tin hieu den Telegram cua nguoi dung."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[INFO] Khong tim thay TELEGRAM_TOKEN hoac TELEGRAM_CHAT_ID trong moi truong. Bo qua buoc gui tin nhan.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url, 
        data=payload, 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print("[INFO] Da gui thong bao tin hieu giao dich qua Telegram Bot!")
    except Exception as e:
        print(f"[ERROR] Khong the gui tin nhan Telegram: {e}")


def main():
    print("==========================================================================")
    print("                PRODUCTION INFERENCE & REBALANCING RUN                    ")
    print(f"                Ngay chay: {datetime.date.today().strftime('%Y-%m-%d')}                    ")
    print("==========================================================================")

    # Thoi gian lay du lieu (1 nam gan nhat de tinh indicators)
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")

    # --- BUOC 1: DOWNLOAD NEWEST DATA ---
    print("\n--- BUOC 1 & 2: GETTING NEWEST DATA & QUALITY CHECK ---")
    prices_dict = {}
    import time
    for symbol in TICKERS:
        try:
            df = get_stock_history(symbol, start_date, end_date)
            if check_data_quality(df, symbol):
                prices_dict[symbol] = df.set_index("time")["close"]
            time.sleep(3.5)  # Tranh chan API (Rate Limit 20 req/min)
        except Exception as e:
            print(f"[WARNING] Loi lay du lieu {symbol}: {e}")
            time.sleep(3.5)
            
    prices_df = pd.DataFrame(prices_dict).sort_index().ffill().dropna()
    print(f"Lay thanh cong du lieu den ngay: {prices_df.index[-1].strftime('%Y-%m-%d')}")

    # --- BUOC 3: INFERENCE (DỰ BÁO) ---
    print("\n--- BUOC 3: GENERATING PREDICTIVE SIGNALS ---")
    
    # Lay du lieu dac trung cho symbol chinh (FPT)
    fpt_history = get_stock_history(PRIMARY_SYMBOL, start_date, end_date)
    fpt_features = add_technical_indicators(fpt_history)
    fpt_features = fpt_features.dropna().reset_index(drop=True)
    
    # Lay dong cuoi cung de dự bao cho ngay mai (index -1)
    latest_row = fpt_features.iloc[[-1]]
    latest_time = latest_row["time"].values[0]
    latest_close = latest_row["close"].values[0]
    
    # Load Random Forest model
    rf_model_path = f"models/{PRIMARY_SYMBOL}_rf.pkl"
    rf_model = joblib.load(rf_model_path)
    
    exclude_cols = ["time", "target", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in fpt_features.columns if c not in exclude_cols]
    
    X_latest_ml = latest_row[feature_cols]
    rf_prob = rf_model.predict_proba(X_latest_ml)[0, 1]
    rf_signal = 1 if rf_prob > 0.5 else 0

    # Load LSTM model
    lstm_scaler_path = f"models/{PRIMARY_SYMBOL}_lstm_scaler.pkl"
    lstm_weights_path = f"models/{PRIMARY_SYMBOL}_lstm.pt"
    
    lstm_meta = joblib.load(lstm_scaler_path)
    scaler = lstm_meta["scaler"]
    seq_len = lstm_meta["seq_len"]
    hidden_dim = lstm_meta["hidden_dim"]
    num_layers = lstm_meta["num_layers"]
    
    # Chuan hoa 10 ngay gan nhat
    X_all_raw = fpt_features[feature_cols].values
    X_scaled = scaler.transform(X_all_raw)
    X_latest_seq = np.array([X_scaled[-seq_len:]])  # Shape: (1, 10, num_features)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMClassifier(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=device))
    lstm_model.eval()
    
    lstm_prob = predict_lstm(lstm_model, X_latest_seq, device)[0]
    lstm_signal = 1 if lstm_prob > 0.5 else 0
    
    print(f"Gia close FPT moi nhat ({pd.to_datetime(latest_time).strftime('%Y-%m-%d')}): {latest_close:,.2f} VND")
    print(f"  [Random Forest] Probability Tang Gia: {rf_prob*100:.2f}% => Tinhieu: {'BUY/HOLD' if rf_signal == 1 else 'CASH/OUT'}")
    print(f"  [LSTM PyTorch]  Probability Tang Gia: {lstm_prob*100:.2f}% => Tinhieu: {'BUY/HOLD' if lstm_signal == 1 else 'CASH/OUT'}")

    # --- BUOC 4: PORTFOLIO OPTIMIZATION ---
    print("\n--- BUOC 4: PORTFOLIO OPTIMIZATION & REBALANCING ---")
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    # Lay ty trong tối uu
    target_weights = optimize_portfolio(mean_returns, cov_matrix, objective="max_sharpe", rf=0.04)
    
    # Load vi the hien tai (Simulate)
    holdings_filepath = "data/current_holdings.json"
    if os.path.exists(holdings_filepath):
        with open(holdings_filepath, "r") as f:
            current_holdings = json.load(f)
    else:
        # Neu chua co file, coi nhu vi the ban dau la chia deu
        current_holdings = {ticker: 1.0 / len(TICKERS) for ticker in TICKERS}
        
    print("\nSo sanh vi the hien tai va vi the toi uu:")
    rebalance_orders = []
    
    for i, ticker in enumerate(TICKERS):
        curr_w = current_holdings.get(ticker, 0.0)
        targ_w = target_weights[i]
        diff_w = targ_w - curr_w
        diff_val = diff_w * PORTFOLIO_VALUE_VND
        
        # Lay gia hien tai cua co phieu
        asset_price = prices_df[ticker].iloc[-1]
        # Quy doi ra so luong co phieu can giao dich (lam tron vi the)
        shares_to_trade = int(diff_val / asset_price)
        
        rebalance_orders.append({
            "ticker": ticker,
            "current_weight": f"{curr_w*100:.1f}%",
            "target_weight": f"{targ_w*100:.1f}%",
            "weight_change": f"{diff_w*100:+.1f}%",
            "action": "BUY" if shares_to_trade > 0 else ("SELL" if shares_to_trade < 0 else "HOLD"),
            "shares": abs(shares_to_trade),
            "estimated_value_vnd": f"{abs(diff_val):,.0f} VND"
        })
        
    orders_df = pd.DataFrame(rebalance_orders)
    print(orders_df.to_string(index=False))

    # --- BUOC 5: ORDER EXECUTION REPORT & TELEGRAM BOT ---
    print("\n--- BUOC 5: ORDER EXECUTION GENERATOR ---")
    
    # Tao tin nhan thong bao
    trade_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    telegram_msg = f"🔔 *TÍN HIỆU GIAO DỊCH HÀNG NGÀY VN-STOCK-QUANT*\n"
    telegram_msg += f"📅 Thời gian chạy: `{trade_time_str}`\n\n"
    telegram_msg += f"📈 *Tín hiệu FPT ngày mai:*\n"
    telegram_msg += f"- RF: `{'BUY/HOLD' if rf_signal == 1 else 'CASH/OUT'}` ({rf_prob*100:.1f}%)\n"
    telegram_msg += f"- LSTM: `{'BUY/HOLD' if lstm_signal == 1 else 'CASH/OUT'}` ({lstm_prob*100:.1f}%)\n\n"
    telegram_msg += f"💼 *Đề xuất điều chỉnh danh mục (Rebalancing):*\n"
    
    for order in rebalance_orders:
        if order["action"] != "HOLD":
            telegram_msg += f"- *{order['action']}* `{order['shares']}` cp *{order['ticker']}* (Thay đổi tỷ trọng: {order['weight_change']}, Giá trị: {order['estimated_value_vnd']})\n"
        else:
            telegram_msg += f"- *HOLD* `{order['ticker']}`\n"
            
    print("\nTin nhan thong bao de xuat thuc thi lenh:")
    print(telegram_msg)
    
    # Gui qua telegram bot (neu da config)
    send_telegram_notification(telegram_msg)
    
    # Luu lai tin hieu giao dich hang ngay ra reports
    os.makedirs("reports", exist_ok=True)
    report_file = "reports/daily_trade_signal.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": trade_time_str,
            "fpt_rf": {"prob": float(rf_prob), "signal": int(rf_signal)},
            "fpt_lstm": {"prob": float(lstm_prob), "signal": int(lstm_signal)},
            "rebalance_orders": rebalance_orders
        }, f, indent=4, ensure_ascii=False)
        
    print(f"\nDa luu tin hieu giao dich ra file JSON tai: {report_file}")
    
    # --- BUOC 6: MOCK PORTFOLIO HOLDINGS UPDATE (Simulate trade execution) ---
    # Trong thuc te, sau khi dat lenh thanh cong, holdings se cap nhat
    new_holdings = {ticker: target_weights[i] for i, ticker in enumerate(TICKERS)}
    with open(holdings_filepath, "w") as f:
        json.dump(new_holdings, f, indent=4)
    print(f"[INFO] Da cap nhat vi the danh muc moi vao file holdings: {holdings_filepath}")


if __name__ == "__main__":
    main()
