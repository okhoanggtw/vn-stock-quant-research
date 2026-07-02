"""
07_pipeline.py
Deploy and maintain ML/DL pipelines in production.
Orchestrator script chay toan bo pipeline quant research tu thu thap du lieu,
tinh toan feature, train model, backtest va toi uu hoa danh muc.
"""

import sys
import os
import subprocess
import time


def run_script(script_name: str):
    """Chay mot python script bang subprocess va in ra trang thai."""
    print("\n" + "=" * 80)
    print(f"   RUNNING PIPELINE STEP: {script_name}   ")
    print("=" * 80)
    
    start_time = time.time()
    # Chay script bang cach goi command line python
    cmd = [sys.executable, script_name]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Read output line by line in real-time
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    duration = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\n=> STEP {script_name} COMPLETED SUCCESSFULLY IN {duration:.2f} SECONDS.")
    else:
        print(f"\n=> ERROR: STEP {script_name} FAILED WITH CODE {process.returncode} IN {duration:.2f} SECONDS.")
        sys.exit(process.returncode)


def main():
    print("==========================================================================")
    print("                   VN STOCK QUANT RESEARCH PIPELINE                       ")
    print("==========================================================================")
    print("Khoi dong pipeline chay nghien cuu va thu nghiem tu dau den cuoi...")
    
    pipeline_start = time.time()
    
    # Buoc 1: Download du lieu raw
    run_script("scripts/01_get_data.py")
    
    # Buoc 2: Tinh toan feature indicators & kiem tra stationarity
    run_script("scripts/02_features.py")
    
    # Buoc 3: Train & Validate model Machine Learning (Random Forest)
    run_script("scripts/03_train_ml.py")
    
    # Buoc 4: Train & Validate model Deep Learning (LSTM PyTorch)
    run_script("scripts/04_train_dl.py")
    
    # Buoc 5: Backtest kiem tra chien luoc & so sanh hieu qua
    run_script("scripts/05_backtest.py")
    
    # Buoc 6: Chay phan bo danh muc dau tu da tai san bang Scipy Optimization
    run_script("scripts/06_portfolio_opt.py")
    
    total_duration = time.time() - pipeline_start
    
    print("\n" + "=" * 80)
    print("                 PIPELINE RUN COMPLETED SUCCESSFULLY                      ")
    print(f"   Tong thoi gian chay: {total_duration:.2f} giay (c.a {total_duration/60:.2f} phut)")
    print("=" * 80)
    print("\nCac file ket qua da duoc luu tru:")
    print(" - Du lieu raw:            data/raw/FPT_raw.csv")
    print(" - Du lieu co dac trung:   data/processed/FPT_features.csv")
    print(" - Model Random Forest:    models/FPT_rf.pkl")
    print(" - Model LSTM PyTorch:     models/FPT_lstm.pt & models/FPT_lstm_scaler.pkl")
    print(" - Bieu do Feature Importance: reports/FPT_feature_importance.png")
    print(" - Bieu do Backtest Equity:   reports/FPT_backtest_equity.png")
    print(" - Bieu do Portfolio Opt:     reports/portfolio_allocation_chart.png")
    print("\nProject da san sang de cap nhat vao CV va mang di phong van!")
    print("=" * 80)


if __name__ == "__main__":
    main()
