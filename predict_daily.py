import os
import pandas as pd
import numpy as np
from keras.models import load_model
from src.preprocessing import DataProcessor

# --- CẤU HÌNH ---
FILE_DATA = 'data/Vinamilk.csv'  # File dữ liệu cập nhật hàng ngày
FILE_MODEL = 'models/vinamilk_lstm.h5'  # Model đã train
LOOK_BACK = 50  # Số ngày quá khứ cần để dự đoán


def predict_next_day():
    # 1. Kiểm tra model và scaler
    if not os.path.exists(FILE_MODEL):
        print("Chưa có model! Hãy chạy main.py trước.")
        return
    
    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        print("Chưa có scaler! Hãy chạy main.py trước.")
        return

    # 2. Đọc dữ liệu mới nhất
    print(">>> Đang đọc dữ liệu thị trường...")
    df = pd.read_csv(FILE_DATA)

    # Xử lý format cột Ngày để tính toán ngày tiếp theo
    try:
        df["Ngày"] = pd.to_datetime(df.Ngày, format="%d/%m/%Y")
    except:
        df["Ngày"] = pd.to_datetime(df.Ngày)

    # Xử lý dữ liệu giá
    cols_price = ['Đóng cửa']
    for col in cols_price:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(',', '').astype(float)

    data_values = df['Đóng cửa'].values.reshape(-1, 1)

    # 3. ✅ FIX: Load scaler đã train (không fit lại!)
    import joblib
    processor = DataProcessor()
    processor.scaler = joblib.load(scaler_path)
    sc_data = processor.transform(data_values)

    # 4. Lấy 50 ngày cuối cùng
    if len(sc_data) < LOOK_BACK:
        print("Dữ liệu không đủ 50 ngày để dự đoán!")
        return

    last_50_days = sc_data[-LOOK_BACK:]
    x_input = last_50_days.reshape(1, LOOK_BACK, 1)

    # 5. Load model và Dự đoán
    print(f">>> Đang load model từ {FILE_MODEL}...")
    model = load_model(FILE_MODEL)

    predicted_scaled = model.predict(x_input)
    predicted_price = processor.inverse_transform(predicted_scaled)
    price_value = predicted_price[0][0]  # Giá dự báo (float)

    # 6. Tạo bảng so sánh (Code của bạn được tích hợp vào đây)

    # Lấy thông tin ngày
    last_date = df['Ngày'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)

    # Lấy giá thực tế ngày cuối cùng
    actual_closing_price = data_values[-1][0]

    # Tạo DataFrame so sánh
    comparison_df = pd.DataFrame({
        'Ngày dự báo': [next_date],
        'Giá dự đoán': [price_value],
        'Giá ngày trước': [actual_closing_price],
        'Chênh lệch': [price_value - actual_closing_price]  # Tính thêm chênh lệch tăng/giảm
    })

    # In kết quả
    print("\n" + "=" * 50)
    print(f"DỮ LIỆU CẬP NHẬT ĐẾN NGÀY: {last_date.strftime('%d/%m/%Y')}")
    print("=" * 50)
    print("BẢNG DỰ BÁO:")
    print(comparison_df.to_string(index=False))  # In bảng đẹp
    print("=" * 50 + "\n")


if __name__ == "__main__":
    predict_next_day()