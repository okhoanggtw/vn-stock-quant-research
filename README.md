# Vietnam Stock Quantitative Research & MLOps Framework

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Apache_Airflow-017A86?style=for-the-badge&logo=apache-airflow&logoColor=white" alt="Apache Airflow">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

Dự án xây dựng và thử nghiệm mô hình học máy (Machine Learning & Deep Learning) dự báo xu hướng giá cổ phiếu rổ VN30, kiểm thử chiến lược giao dịch (Backtesting), tối ưu hóa danh mục đầu tư (Portfolio Optimization) và vận hành tự động hóa (MLOps).

Dự án được phát triển dưới dạng một **Production-ready Framework**, mô phỏng chính xác quy trình nghiên cứu định lượng (Quantitative Research) và vận hành hệ thống giao dịch tự động tại các quỹ đầu tư chuyên nghiệp.

---

## 🎯 1. Bài Toán Thực Tế Dự Án Giải Quyết

Trong đầu tư tài chính, việc đưa ra quyết định giao dịch và phân bổ vốn thường đối mặt với các vấn đề lớn:
1.  **Rủi ro từ tâm lý & trực giác**: Quyết định mua/bán theo cảm tính hoặc đám đông thường dẫn đến thua lỗ. Dự án giải quyết bằng cách áp dụng mô hình toán học và học máy để đưa ra tín hiệu giao dịch dựa trên dữ liệu lịch sử khách quan.
2.  **Rò rỉ dữ liệu trong kiểm thử (Look-ahead Bias/Data Leakage)**: Nhiều nghiên cứu học thuật bị ảo tưởng về hiệu quả do chia tập dữ liệu ngẫu nhiên (dùng dữ liệu tương lai đoán quá khứ). Dự án khắc phục triệt để bằng cách áp dụng **Sequential Split** và **TimeSeriesSplit Cross-Validation**.
3.  **Chi phí giao dịch bị bỏ qua**: Backtest thông thường không tính phí giao dịch khiến tần suất trade quá dày (overtrading) bào mòn tài khoản. Dự án tích hợp khấu trừ **0.15% thuế phí thực tế** cho mỗi giao dịch tại thị trường Việt Nam.
4.  **Phân bổ vốn phi khoa học**: Việc chia đều vốn (Equal Weight) không tối ưu hóa được tỷ lệ Lợi nhuận/Rủi ro. Dự án sử dụng mô hình tối ưu **Markowitz Mean-Variance** để tìm ra tỷ trọng tối đa hóa Sharpe Ratio của danh mục.
5.  **Thiếu tính tự động hóa (MLOps)**: Khoảng cách giữa mô hình nghiên cứu (Jupyter Notebook) và vận hành thực tế rất lớn. Dự án đóng gói toàn bộ quy trình thành các **Task định thời trên Apache Airflow** và container hóa bằng **Docker** để chạy tự động hàng ngày.

---

## 🛠️ 2. Tại Sao Sử Dụng Các Công Cụ Này? (Tech Stack Justification)

*   **Python (Pandas, NumPy)**: Ngôn ngữ tiêu chuẩn trong lĩnh vực tài chính định lượng. Pandas dùng để xử lý dữ liệu bảng (Time-Series) và NumPy dùng tính toán nhanh trên ma trận hiệp phương sai.
*   **Ta (Technical Analysis Library)**: Thư viện giúp tính toán nhanh các chỉ báo kỹ thuật chuẩn hóa (RSI, MACD, Bollinger Bands, ATR) thay vì tự code lại công thức, đảm bảo tính chính xác tuyệt đối.
*   **Scikit-Learn**: Cung cấp các mô hình Baseline ML (`RandomForest`, `GradientBoosting`) và đặc biệt là công cụ chéo hóa chuỗi thời gian `TimeSeriesSplit`.
*   **PyTorch**: Xây dựng mạng nơ-ron hồi quy **LSTM (Long Short-Term Memory)** để nắm bắt các đặc trưng chuỗi thời gian dài hạn tốt hơn các mô hình học máy truyền thống.
*   **SciPy (Optimization)**: Sử dụng trực tiếp solver phi tuyến `SLSQP` trong `scipy.optimize` để tự viết giải thuật tối ưu hóa danh mục đầu tư với các ràng buộc thực tế, thay vì sử dụng các thư viện black-box đóng gói sẵn.
*   **Apache Airflow & Docker**: Sử dụng để điều phối pipeline (Orchestration). Airflow quản lý thứ tự chạy của các tác vụ dưới dạng DAG và tự động gửi cảnh báo khi lỗi. Docker đảm bảo hệ thống có thể deploy lên bất kỳ server VPS nào (như Google Cloud) mà không bị lỗi khác biệt môi trường.
*   **Telegram Bot API**: Cầu nối đẩy tín hiệu giao dịch thực tế hàng ngày trực tiếp về điện thoại người dùng, mô phỏng quá trình sinh lệnh tự động (Execution).

---

## 📂 3. Cấu Trúc Hệ Thống

```
vn-stock-quant-research/
├── src/                      # Module cốt lõi tái sử dụng (Core API)
│   ├── data_loader.py        # Tải dữ liệu lịch sử từ vnstock API
│   ├── features.py           # Tính các chỉ báo kỹ thuật, ADF test, lag features
│   ├── models.py             # Định nghĩa LSTM PyTorch & mô hình ML, TimeSeriesSplit CV
│   ├── backtest.py           # Bộ tính toán hiệu suất chiến lược và backtesting
│   └── portfolio.py          # Bộ giải toán tối ưu hóa danh mục đầu tư bằng Scipy
├── scripts/                  # Các script thực thi theo từng bước trong pipeline
│   ├── 01_get_data.py        # Tải dữ liệu lịch sử FPT & vẽ đồ thị giá
│   ├── 02_features.py        # Tiền xử lý dữ liệu, kiểm định tính dừng (ADF Test)
│   ├── 03_train_ml.py        # Huấn luyện mô hình Random Forest & xuất Feature Importance
│   ├── 04_train_dl.py        # Huấn luyện mô hình LSTM bằng PyTorch & chuẩn hóa dữ liệu
│   ├── 05_backtest.py        # So sánh chiến lược RF vs LSTM vs Buy-and-Hold trên tập Test
│   ├── 06_portfolio_opt.py   # Chạy tối ưu danh mục rổ VN30 (Max Sharpe, Min Var)
│   ├── 07_pipeline.py        # Chạy pipeline thử nghiệm end-to-end local nhanh
│   └── 08_production_inference.py # Script chạy live rebalance hàng ngày & Telegram Alert
├── dags/                     # Apache Airflow DAG (Lập lịch tự động)
│   └── vn_stock_quant_dag.py # File điều phối chạy tự động 15:30 từ thứ 2 đến thứ 6
├── data/                     # Thư mục lưu trữ dữ liệu thô, processed và holdings hiện tại
├── models/                   # Lưu trữ các file checkpoints mô hình (.pkl, .pt)
├── reports/                  # Xuất các biểu đồ trực quan và file tín hiệu hàng ngày
├── Dockerfile                # Đóng gói container ứng dụng Python
├── Dockerfile.airflow        # Đóng gói container Apache Airflow chứa thư viện mô hình
├── docker-compose.yml        # Khởi chạy cụm dịch vụ Airflow Local (Postgres, Webserver, Scheduler)
├── .gitignore                # Quản lý các tệp tin loại trừ khỏi Git
└── requirements.txt          # Khai báo thư viện phụ thuộc
```

---

## 🚀 4. Hướng Dẫn Sử Dụng Chi Tiết

### Lựa chọn 1: Chạy cục bộ bằng môi trường ảo (Local Development)
1.  **Cài đặt môi trường**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
2.  **Chạy toàn bộ pipeline thử nghiệm**:
    ```bash
    python scripts/07_pipeline.py
    ```

### Lựa chọn 2: Đóng gói và chạy bằng Docker
Bạn có thể build container và chạy pipeline mà không cần cài đặt Python thủ công:
```bash
# Build Image
docker build -t vn-stock-quant-app .

# Chạy Pipeline
docker run --rm -v $(pwd):/app vn-stock-quant-app
```

### Lựa chọn 3: Triển khai vận hành thực tế bằng Apache Airflow
Chạy cụm dịch vụ Apache Airflow bằng Docker Compose:
1.  **Thiết lập biến môi trường Telegram** (Tạo file `.env` ở thư mục gốc):
    ```env
    TELEGRAM_TOKEN=8783233904:AAH03k_4TT8TGFMowekxKLuXc66AaV14X6c
    TELEGRAM_CHAT_ID=5755999444
    ```
2.  **Khởi động các dịch vụ**:
    ```bash
    docker-compose up --build -d
    ```
3.  **Truy cập Airflow Web UI**:
    *   Mở trình duyệt: `http://localhost:8080` (hoặc `http://<IP_VPS>:8080` nếu chạy trên Google Cloud).
    *   Tài khoản đăng nhập mặc định: **`admin`** / Mật khẩu: **`admin`**.
    *   Bật DAG **`vn_stock_quant_research_pipeline`** sang trạng thái **ON** để lập lịch tự động lúc **15:30 chiều hàng ngày** từ Thứ 2 đến Thứ 6.

---

## 📊 5. Kết Quả Thực Nghiệm & Báo Cáo

### 1. Hiệu Suất Chiến Lược Giao Dịch (Backtest FPT)
*Dữ liệu kiểm thử ngoài mẫu (Out-of-sample Test): **03/01/2025 đến 23/06/2025** (113 ngày giao dịch), khấu trừ **0.15% thuế phí**.*

| Chỉ Số (Annualized) | Mua & Nắm Giữ (Buy & Hold) | Chiến Lược Random Forest | Chiến Lược LSTM (PyTorch) |
| :--- | :---: | :---: | :---: |
| **Tổng Lợi Nhuận** | -21.55% | -19.75% | **-13.97%** |
| **Lợi Nhuận Năm** | -41.80% | -38.77% | **-28.50%** |
| **Sharpe Ratio** | -1.458 | -1.483 | **-1.043** |
| **Sortino Ratio** | -1.848 | -1.639 | **-1.280** |
| **Sụt Giảm Tối Đa (Max DD)** | -31.88% | **-25.97%** | -28.47% |
| **Hệ Số Lợi Nhuận (Profit Factor)**| 0.76 | 0.72 | **0.81** |

> **Nhận xét**: Mô hình LSTM PyTorch giúp bảo vệ tài sản (downside protection) rất tốt trong giai đoạn thị trường điều chỉnh mạnh vào đầu năm 2025.

### 2. Tối Ưu Hóa Danh Mục (Portfolio Optimization với 30 mã VN30)
*Tối ưu hóa danh mục gồm 30 tài sản VN30 bằng solver SLSQP (Scipy) với lãi suất phi rủi ro $r_f = 4.0\%$.*

*   **Chỉ số hiệu suất danh mục sau tối ưu:**
    -   **Expected Return (Kỳ vọng lợi nhuận năm):** Equal Weight: 26.84% | **Max Sharpe: 39.10%** | Min Variance: 9.03%
    -   **Volatility (Rủi ro biến động năm):** Equal Weight: 22.28% | Max Sharpe: 23.90% | **Min Variance: 18.10%**
    -   **Sharpe Ratio (Annualized):** Equal Weight: 1.025 | **Max Sharpe: 1.468** | Min Variance: 0.278

*Các biểu đồ trực quan chi tiết được lưu trong thư mục `reports/`:*
-   `reports/FPT_feature_importance.png`: Đồ thị độ quan trọng đặc trưng.
-   `reports/FPT_backtest_equity.png`: So sánh biểu đồ tài sản lũy kế.
-   `reports/portfolio_allocation_chart.png`: Đồ thị phân bổ tỷ trọng tối ưu danh mục.
