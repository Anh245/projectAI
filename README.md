# 📈 Dự Đoán Giá Cổ Phiếu Vinamilk với LSTM

Dự án sử dụng mạng LSTM (Long Short-Term Memory) để dự đoán giá cổ phiếu Vinamilk dựa trên dữ liệu lịch sử.

## 🎯 Tính Năng

- **Train Model**: Huấn luyện mô hình LSTM với dữ liệu lịch sử
- **Đánh Giá Model**: Tính toán các chỉ số R2, MAE, MAPE
- **Dự Đoán Hàng Ngày**: Dự báo giá cổ phiếu cho ngày tiếp theo
- **Chống Overfitting**: Áp dụng dropout, early stopping, validation split

## 📁 Cấu Trúc Dự Án

```
├── Data/
│   └── Vinamilk.csv          # Dữ liệu giá cổ phiếu
├── models/
│   ├── vinamilk_lstm.h5      # Model đã train
│   └── scaler.pkl            # Scaler để chuẩn hóa dữ liệu
├── src/
│   ├── data_loader.py        # Load và làm sạch dữ liệu
│   ├── preprocessing.py      # Xử lý và chuẩn hóa dữ liệu
│   ├── model_builder.py      # Kiến trúc mô hình LSTM
│   └── visualization.py      # Vẽ biểu đồ kết quả
├── main.py                   # Train model
├── run_evaluation.py         # Đánh giá model
├── predict_daily.py          # Dự đoán ngày tiếp theo
└── requirements.txt          # Thư viện cần thiết
```

## 🚀 Cài Đặt

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt file dữ liệu `Vinamilk.csv` vào thư mục `Data/` với các cột:
- Ngày (định dạng: dd/mm/yyyy)
- Đóng cửa
- Mở cửa
- Cao nhất
- Thấp nhất

## 📖 Hướng Dẫn Sử Dụng

### Bước 1: Huấn luyện Model

```bash
python main.py
```

**Kết quả:**
- Tạo file `models/vinamilk_lstm.h5` (model đã train)
- Tạo file `models/scaler.pkl` (scaler để chuẩn hóa)
- Hiển thị các chỉ số đánh giá (R2, MAE, MAPE)
- Vẽ biểu đồ so sánh giá thực tế vs dự đoán

**Lưu ý:** Model sử dụng:
- 1500 ngày đầu làm tập train
- Phần còn lại làm tập test
- Validation split 20% để chống overfitting
- Early stopping với patience=10

### Bước 2: Đánh Giá Model (Không cần train lại)

```bash
python run_evaluation.py
```

**Kết quả:**
- Load model đã train
- Tính toán lại các chỉ số đánh giá
- Vẽ biểu đồ so sánh

### Bước 3: Dự Đoán Ngày Tiếp Theo

```bash
python predict_daily.py
```

**Kết quả:**
```
==================================================
DỮ LIỆU CẬP NHẬT ĐẾN NGÀY: 15/12/2024
==================================================
BẢNG DỰ BÁO:
Ngày dự báo  Giá dự đoán  Giá ngày trước  Chênh lệch
2024-12-16      85000.0        84500.0        500.0
==================================================
```

## 🛠️ Cấu Hình

### Thay đổi tham số trong code:

**`main.py`, `run_evaluation.py`, `predict_daily.py`:**
```python
LOOK_BACK = 50        # Số ngày quá khứ để dự đoán
SPLIT_INDEX = 1500    # Số dòng dữ liệu cho tập train
```

**`src/model_builder.py`:**
```python
# Kiến trúc LSTM
LSTM(units=64)        # Số neurons lớp 1
LSTM(units=32)        # Số neurons lớp 2
Dropout(0.3)          # Tỷ lệ dropout
```

**`main.py` - Training:**
```python
epochs=100            # Số epoch tối đa
batch_size=50         # Kích thước batch
validation_split=0.2  # Tỷ lệ validation
patience=10           # Early stopping patience
```

## 📊 Các Chỉ Số Đánh Giá

- **R2 Score**: Độ phù hợp của model (càng gần 1 càng tốt)
- **MAE**: Sai số tuyệt đối trung bình (VNĐ)
- **MAPE**: Phần trăm sai số tuyệt đối trung bình (%)

## ⚠️ Lưu Ý Quan Trọng

### Tránh Overfitting (Học Vẹt)

Dự án đã áp dụng các kỹ thuật chống overfitting:

1. **Fit scaler chỉ trên train set** - Tránh data leakage
2. **Validation split 20%** - Theo dõi overfitting trong quá trình train
3. **Early stopping** - Dừng train khi val_loss không giảm
4. **Dropout layers** - Giảm overfitting trong mạng neural
5. **Giảm model complexity** - Sử dụng 64 và 32 units thay vì 128 và 64

### Quy Trình Đúng

✅ **ĐÚNG:**
```python
# Train: Fit scaler trên train set
train_scaled = processor.fit_transform(train_data)
test_scaled = processor.transform(test_data)

# Predict: Load scaler đã lưu
processor.scaler = joblib.load('models/scaler.pkl')
data_scaled = processor.transform(new_data)
```

❌ **SAI:**
```python
# Fit scaler trên toàn bộ dữ liệu (bao gồm test set)
all_scaled = processor.fit_transform(all_data)  # Data leakage!
```

## 🔄 Cập Nhật Dữ Liệu Mới

1. Thêm dữ liệu mới vào file `Data/Vinamilk.csv`
2. Chạy `python predict_daily.py` để dự đoán ngày tiếp theo
3. Nếu muốn train lại model với dữ liệu mới: `python main.py`

## 📝 Yêu Cầu Hệ Thống

- Python 3.7+
- TensorFlow/Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Joblib
