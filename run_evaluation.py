import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# Import các module từ dự án của bạn
from src.data_loader import load_and_clean_data
from src.preprocessing import DataProcessor
from src.visualization import plot_predictions

# --- CẤU HÌNH ---
FILE_PATH = os.path.join('data', 'Vinamilk.csv')
MODEL_PATH = os.path.join('models', 'vinamilk_lstm.h5')  # Đường dẫn đến file model đã train
LOOK_BACK = 50
SPLIT_INDEX = 1500


def run_eval():
    # 1. Kiểm tra xem đã có model và scaler chưa
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy file model tại '{MODEL_PATH}'")
        print("Bạn cần chạy file main.py trước để huấn luyện và lưu model.")
        return
    
    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"LỖI: Không tìm thấy file scaler tại '{scaler_path}'")
        print("Bạn cần chạy file main.py trước để lưu scaler.")
        return

    # 2. Load dữ liệu
    print(">>> Đang đọc dữ liệu...")
    df = load_and_clean_data(FILE_PATH)
    df_close = df[['Đóng cửa']].copy()
    df_close.index = df['Ngày']
    data_values = df_close.values

    # 3. ✅ FIX: Load scaler đã train (giống như lúc train)
    import joblib
    processor = DataProcessor()
    processor.scaler = joblib.load(scaler_path)
    
    # Chia train/test giống như lúc train
    train_data_raw = data_values[:SPLIT_INDEX]
    test_data_raw = data_values[SPLIT_INDEX:]
    
    # Transform bằng scaler đã load
    train_scaled = processor.transform(train_data_raw)
    test_scaled = processor.transform(test_data_raw)

    # Chuẩn bị dữ liệu Train
    x_train, y_train = processor.create_dataset(train_scaled, LOOK_BACK)
    
    # Chuẩn bị dữ liệu Test
    combined_data = np.vstack([train_scaled[-LOOK_BACK:], test_scaled])
    x_test, y_test = processor.create_dataset(combined_data, LOOK_BACK)

    # 4. Load Model đã lưu (BỎ QUA BƯỚC TRAIN)
    print(f">>> Đang load model từ: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 5. Thực hiện dự đoán
    print(">>> Đang tính toán...")
    # Dự đoán Train
    y_train_predict = model.predict(x_train)
    y_train_predict = processor.inverse_transform(y_train_predict)
    y_train_actual = processor.inverse_transform(y_train)

    # Dự đoán Test
    y_test_predict = model.predict(x_test)
    y_test_predict = processor.inverse_transform(y_test_predict)
    y_test_actual = data_values[SPLIT_INDEX:]

    # Cắt dữ liệu cho khớp độ dài (phòng trường hợp lệch 1-2 dòng)
    min_len = min(len(y_test_actual), len(y_test_predict))
    y_test_actual = y_test_actual[:min_len]
    y_test_predict = y_test_predict[:min_len]

    # 6. In các chỉ số đánh giá (Code bạn yêu cầu)
    print("\n" + "=" * 40)
    print("KẾT QUẢ ĐÁNH GIÁ (KHÔNG CẦN TRAIN LẠI)")
    print("=" * 40)

    # Đánh giá tập Train
    print('\n--- TẬP TRAIN ---')
    print('Độ phù hợp (R2):', r2_score(y_train_actual, y_train_predict))
    print('Sai số tuyệt đối (MAE):', mean_absolute_error(y_train_actual, y_train_predict))
    print('Phần trăm sai số (MAPE):', mean_absolute_percentage_error(y_train_actual, y_train_predict))

    # Đánh giá tập Test
    print('\n--- TẬP TEST (QUAN TRỌNG) ---')
    print('Độ phù hợp (R2):', r2_score(y_test_actual, y_test_predict))
    print('Sai số tuyệt đối (MAE):', mean_absolute_error(y_test_actual, y_test_predict))
    print('Phần trăm sai số (MAPE):', mean_absolute_percentage_error(y_test_actual, y_test_predict))
    print("\n" + "=" * 40)
    # R2-score
    print('Độ phù hợp tập test:', r2_score(y_test_actual, y_test_predict))

    # MAE
    print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', mean_absolute_error(y_test_actual, y_test_predict))

    # MAPE
    print('Phần trăm sai số tuyệt đối trung bình tập test:',
          mean_absolute_percentage_error(y_test_actual, y_test_predict))

    # --------------------------------------------------------
    print("\n" + "=" * 40)

    # 7. Vẽ biểu đồ
    plot_predictions(df_close, y_train_predict, y_test_predict, LOOK_BACK, SPLIT_INDEX)


if __name__ == "__main__":
    run_eval()