import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from keras.callbacks import ModelCheckpoint

# Import các module tự viết
from src.data_loader import load_and_clean_data
from src.preprocessing import DataProcessor
from src.model_builder import build_lstm_model
from src.visualization import plot_predictions

# --- CẤU HÌNH ---
FILE_PATH = os.path.join('Data', 'Vinamilk.csv')  # Đường dẫn file dữ liệu
MODEL_PATH = os.path.join('models', 'vinamilk_lstm.h5')
LOOK_BACK = 20  # Giảm từ 50 xuống 20 để model dễ học hơn
SPLIT_INDEX = None  # Sẽ tính toán dựa trên 80% dữ liệu


def main():
    # 1. Load dữ liệu
    print(">>> Đang đọc dữ liệu...")
    df = load_and_clean_data(FILE_PATH)

    # Tạo dataframe chỉ chứa giá đóng cửa để xử lý
    df_close = df[['Đóng cửa']].copy()
    df_close.index = df['Ngày']  # Set index là ngày để vẽ biểu đồ cho đẹp
    data_values = df_close.values

    # 2. Chia train/test & Chuẩn hóa
    print(">>> Đang xử lý dữ liệu...")
    
    # Tính toán SPLIT_INDEX dựa trên 80% dữ liệu cho training
    total_rows = len(data_values)
    SPLIT_INDEX = int(total_rows * 0.8)
    print(f"Tổng số dòng dữ liệu: {total_rows}")
    print(f"Training: {SPLIT_INDEX} dòng, Testing: {total_rows - SPLIT_INDEX} dòng")
    
    train_data_raw = data_values[:SPLIT_INDEX]
    test_data_raw = data_values[SPLIT_INDEX:]

    processor = DataProcessor()
    # ✅ FIX: Chỉ fit scaler trên tập TRAIN (tránh data leakage)
    train_scaled = processor.fit_transform(train_data_raw)
    
    # Transform test data bằng scaler đã fit
    test_scaled = processor.transform(test_data_raw)
    
    # Tạo chuỗi dữ liệu (Sliding Window) cho Train
    x_train, y_train = processor.create_dataset(train_scaled, LOOK_BACK)

    # Tạo chuỗi dữ liệu cho Test
    # Cần LOOK_BACK ngày cuối của train để dự đoán ngày đầu tiên của test
    combined_data = np.vstack([train_scaled[-LOOK_BACK:], test_scaled])
    x_test, y_test = processor.create_dataset(combined_data, LOOK_BACK)
    
    # ✅ DEBUG: Kiểm tra kích thước
    print(f"Train data: {len(train_data_raw)} samples")
    print(f"Test data: {len(test_data_raw)} samples") 
    print(f"X_test shape: {x_test.shape}")
    print(f"Y_test shape: {y_test.shape}")
    print(f"Expected test samples: {len(test_data_raw)}")
    
    # ✅ FIX: Sử dụng dữ liệu test đã scaled để tránh data leakage
    y_test_actual = processor.inverse_transform(y_test)

    # 3. Xây dựng và Huấn luyện Model
    print(">>> Đang xây dựng và huấn luyện Model...")
    model = build_lstm_model((x_train.shape[1], 1))

    # ✅ FIX: Thêm validation split và early stopping
    from keras.callbacks import EarlyStopping
    
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    # ✅ Thêm ReduceLROnPlateau để giảm learning rate khi model không cải thiện
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)

    # ✅ FIX: Chia validation theo thời gian thay vì random
    # Lấy 20% cuối của training data làm validation (theo thời gian)
    val_split_idx = int(len(x_train) * 0.8)
    x_train_final = x_train[:val_split_idx]
    y_train_final = y_train[:val_split_idx]
    x_val = x_train[val_split_idx:]
    y_val = y_train[val_split_idx:]
    
    print(f"Final training: {len(x_train_final)} samples")
    print(f"Validation: {len(x_val)} samples")
    
    # Train với validation data riêng biệt
    # ✅ Cải thiện: Giảm batch_size và tăng epochs để model học tốt hơn
    model.fit(x_train_final, y_train_final, epochs=300, batch_size=16, 
              validation_data=(x_val, y_val), verbose=1, 
              callbacks=[checkpoint, early_stop, reduce_lr])

    # 4. Dự báo và Đánh giá
    print(">>> Đang đánh giá model...")
    # Dự báo trên tập Train (chỉ phần final training, không bao gồm validation)
    y_train_predict = model.predict(x_train_final)
    y_train_predict = processor.inverse_transform(y_train_predict)
    y_train_actual = processor.inverse_transform(y_train_final)

    # Dự báo trên tập Test
    y_test_predict = model.predict(x_test)
    y_test_predict = processor.inverse_transform(y_test_predict)

    # Tính chỉ số đánh giá (trên tập Test)
    # Lưu ý: cắt y_test_actual cho khớp độ dài với y_test_predict
    min_len = min(len(y_test_actual), len(y_test_predict))
    y_test_actual = y_test_actual[:min_len]
    y_test_predict = y_test_predict[:min_len]

    print(f"R2 Score (Test): {r2_score(y_test_actual, y_test_predict)}")
    print(f"MAE (Test): {mean_absolute_error(y_test_actual, y_test_predict)}")
    print(f"MAPE (Test): {mean_absolute_percentage_error(y_test_actual, y_test_predict)}")

    # 5. Vẽ biểu đồ
    # Tính toán lại split index cho visualization vì chỉ dùng final training data
    final_train_end_idx = LOOK_BACK + len(y_train_predict)
    plot_predictions(df_close, y_train_predict, y_test_predict, LOOK_BACK, final_train_end_idx)

    # 6. Lưu scaler để dùng cho predict_daily.py
    import joblib
    joblib.dump(processor.scaler, 'models/scaler.pkl')
    print(">>> Đã lưu scaler vào models/scaler.pkl")
    
    # 7. Dự đoán ngày mai (Next Day)
    print(">>> Dự đoán ngày tiếp theo...")
    
    # ✅ FIX: Dùng toàn bộ dữ liệu đã scaled đúng cách
    all_data_scaled = np.vstack([train_scaled, test_scaled])
    last_window = all_data_scaled[-LOOK_BACK:]
    last_window = last_window.reshape(1, LOOK_BACK, 1)

    next_day_pred = model.predict(last_window)
    next_day_price = processor.inverse_transform(next_day_pred)

    last_date = df['Ngày'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)

    print(f"Ngày cuối cùng trong dữ liệu: {last_date.strftime('%d/%m/%Y')}")
    print(f"Dự báo giá ngày {next_date.strftime('%d/%m/%Y')}: {next_day_price[0][0]:,.0f} VNĐ")
    
    # ✅ DEBUG: Kiểm tra khoảng thời gian dữ liệu
    print(f"\n=== THÔNG TIN DỮ LIỆU ===")
    print(f"Ngày đầu tiên: {df['Ngày'].min().strftime('%d/%m/%Y')}")
    print(f"Ngày cuối cùng: {df['Ngày'].max().strftime('%d/%m/%Y')}")
    print(f"Tổng số ngày: {len(df)}")
    print(f"Training data: {SPLIT_INDEX} ngày")
    print(f"Test data: {len(df) - SPLIT_INDEX} ngày")


if __name__ == "__main__":
    # Tạo thư mục models nếu chưa có
    if not os.path.exists('models'):
        os.makedirs('models')
    main()