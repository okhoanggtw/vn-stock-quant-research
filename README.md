# VN Stock Quant Research

Du an nghien cuu du doan xu huong gia co phieu Viet Nam bang Machine Learning,
ket hop technical indicators va backtest chien luoc trading.

> Status: Dang xay dung (Ngay 1/10)

## Muc tieu
- Xay dung pipeline: thu thap du lieu -> feature engineering -> model ML/DL -> backtest -> danh gia
- Demo quy trinh quant research co ban tren thi truong chung khoan VN

## Cau truc project
```
vn-stock-quant-research/
├── src/                # Core modules tai su dung
├── scripts/            # Script chay theo tung buoc (01, 02, 03...)
├── data/raw/           # Du lieu goc
├── data/processed/     # Du lieu da xu ly, sang feature
├── models/             # Model da train
├── reports/            # Chart, ket qua backtest
└── requirements.txt
```

## Cach chay

```bash
pip install -r requirements.txt
python scripts/01_get_data.py
```~

## Tien do

- [x] Ngay 1: Lay du lieu gia co phieu (vnstock), EDA co ban
- [ ] Ngay 2: Tinh technical indicators (RSI, MACD, MA, Bollinger Bands)
- [ ] Ngay 3-4: Xay dung model Logistic Regression du doan xu huong
- [ ] Ngay 5: Danh gia model (accuracy, confusion matrix, feature importance)
- [ ] Ngay 6: Chuyen prediction thanh trading signal
- [ ] Ngay 7: Backtest chien luoc, so sanh voi buy-and-hold
- [ ] Ngay 8: Model LSTM (PyTorch)
- [ ] Ngay 9: Hoan thien report
- [ ] Ngay 10: Review tong the

## Tech stack
Python, pandas, scikit-learn, PyTorch, pandas-ta, vnstock
