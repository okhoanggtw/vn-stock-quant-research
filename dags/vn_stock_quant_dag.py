"""
vn_stock_quant_dag.py
Apache Airflow DAG de dieu phoi toan bo pipeline Quant Research va Live Trading.
Chay tu dong vao 15:30 chieu tu Thu 2 den Thu 6 hang tuan.
"""

from datetime import datetime, timedelta
from airflow import DAG
# pyrefly: ignore [missing-import]
from airflow.operators.bash import BashOperator
# pyrefly: ignore [missing-import]
from airflow.utils.dates import days_ago

# Cau hinh mac dinh cho cac Task
default_args = {
    "owner": "quant_researcher",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Khoi tao DAG
with DAG(
    "vn_stock_quant_research_pipeline",
    default_args=default_args,
    description="Pipeline tu dong tai du lieu, train model, backtest va thuc thi rebalance danh muc.",
    schedule_interval="30 15 * * 1-5",  # Chạy lúc 15:30 từ thứ 2 đến thứ 6
    start_date=days_ago(1),
    catchup=False,
    tags=["quant", "trading", "mlops"],
) as dag:

    # Task 1: Lay du lieu raw hang ngay tu vnstock API
    task_get_data = BashOperator(
        task_id="fetch_raw_stock_data",
        bash_command="python scripts/01_get_data.py",
    )

    # Task 2: Xu ly data, tao chi bao ky thuat va kiem tra tinh dung (ADF Test)
    task_features = BashOperator(
        task_id="feature_engineering_and_adf_test",
        bash_command="python scripts/02_features.py",
    )

    # Task 3: Train model Machine Learning (Random Forest) va xuat feature importance
    task_train_ml = BashOperator(
        task_id="train_machine_learning_model",
        bash_command="python scripts/03_train_ml.py",
    )

    # Task 4: Train model Deep Learning (LSTM PyTorch)
    task_train_dl = BashOperator(
        task_id="train_deep_learning_lstm_model",
        bash_command="python scripts/04_train_dl.py",
    )

    # Task 5: Chay backtest kiem tra chien luoc giao dich va ve bieu do Equity
    task_backtest = BashOperator(
        task_id="backtest_trading_strategies",
        bash_command="python scripts/05_backtest.py",
    )

    # Task 6: Tối uu hoa danh muc đa tai san bang Scipy SLSQP Solver
    task_portfolio = BashOperator(
        task_id="portfolio_optimization_scipy",
        bash_command="python scripts/06_portfolio_opt.py",
    )

    # Task 7: Chay live inference, sinh quyet dinh rebalance va gui Telegram notification
    task_live_inference = BashOperator(
        task_id="live_inference_and_telegram_alert",
        bash_command="python scripts/08_production_inference.py",
    )

    # Thiet lap luong phu thuoc giua cac Task (DAG Flow)
    task_get_data >> task_features
    task_features >> [task_train_ml, task_train_dl] >> task_backtest
    task_backtest >> task_portfolio >> task_live_inference
