import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(original_df, train_predict, test_predict, look_back, split_index):
    """
    Vẽ biểu đồ so sánh giá thực, giá train dự đoán và giá test dự đoán.
    """
    # Tính toán kích thước thực tế dựa trên dữ liệu đầu vào
    train_predict_flat = train_predict.flatten() if train_predict.ndim > 1 else train_predict
    test_predict_flat = test_predict.flatten() if test_predict.ndim > 1 else test_predict
    
    # Tính toán index dates dựa trên kích thước thực tế của predictions
    # Train predict bắt đầu từ look_back và có độ dài = len(train_predict)
    train_start_idx = look_back
    train_end_idx = train_start_idx + len(train_predict_flat)
    
    # Test predict bắt đầu từ split_index
    test_start_idx = split_index
    test_end_idx = test_start_idx + len(test_predict_flat)
    
    # ✅ FIX: Đảm bảo test không vượt quá dữ liệu thực tế
    available_test_days = len(original_df) - test_start_idx
    if len(test_predict_flat) > available_test_days:
        print(f"WARNING: Test predict có {len(test_predict_flat)} samples nhưng chỉ có {available_test_days} ngày thực tế")
        test_predict_flat = test_predict_flat[:available_test_days]
        test_end_idx = len(original_df)
    
    # Đảm bảo không vượt quá kích thước của original_df
    train_end_idx = min(train_end_idx, len(original_df))
    test_end_idx = min(test_end_idx, len(original_df))
    
    # ✅ DEBUG: In thông tin để kiểm tra
    print(f"\n=== DEBUG VISUALIZATION ===")
    print(f"Original data range: {original_df.index[0]} to {original_df.index[-1]}")
    print(f"Train predict: {len(train_predict_flat)} samples, dates {train_start_idx} to {train_end_idx}")
    print(f"Test predict: {len(test_predict_flat)} samples, dates {test_start_idx} to {test_end_idx}")
    print(f"Split index: {split_index}")
    
    # Lấy dates tương ứng
    train_dates = original_df.index[train_start_idx:train_end_idx]
    test_dates = original_df.index[test_start_idx:test_end_idx]
    
    # Cắt predictions cho khớp với dates (phòng trường hợp lệch)
    train_predict_plot = train_predict_flat[:len(train_dates)]
    test_predict_plot = test_predict_flat[:len(test_dates)]

    plt.figure(figsize=(15, 6))

    # Vẽ giá thực
    plt.plot(original_df.index, original_df['Đóng cửa'], label='Giá thực tế', color='red', linewidth=1.5)

    # Vẽ dự đoán Train
    plt.plot(train_dates, train_predict_plot, label='Dự đoán (Train)', color='green', linewidth=1)

    # Vẽ dự đoán Test
    plt.plot(test_dates, test_predict_plot, label='Dự đoán (Test)', color='blue', linewidth=1)

    plt.title('Biểu đồ so sánh giá thực tế và dự báo Vinamilk (VNM)')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá (VNĐ)')
    plt.legend()
    
    # ✅ Giới hạn trục x để không hiển thị quá xa
    plt.xlim(original_df.index[0], original_df.index[-1])
    
    # Xoay nhãn ngày tháng cho dễ đọc
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()