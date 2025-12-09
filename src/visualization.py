import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(original_df, train_predict, test_predict, look_back, split_index):
    """
    Vẽ biểu đồ so sánh giá thực, giá train dự đoán và giá test dự đoán.
    """
    # Chuẩn bị dữ liệu để vẽ
    # Chúng ta cần align (căn chỉnh) lại index của dự đoán với ngày tháng thực tế

    # Tạo mảng rỗng để chứa dữ liệu vẽ
    train_predict_plot = pd.DataFrame(index=original_df.index)
    train_predict_plot['Train Predict'] = None

    # Gán dữ liệu train (lưu ý offset do look_back)
    # Dữ liệu train bắt đầu từ dòng thứ `look_back`
    train_end_idx = split_index
    # Cắt dataframe theo index tương ứng

    # Đơn giản hóa việc vẽ bằng cách dùng index ngày tháng từ df gốc
    # Train predict tương ứng với đoạn: original_df[look_back : split_index]
    train_dates = original_df.index[look_back:split_index]
    test_dates = original_df.index[split_index:]

    plt.figure(figsize=(15, 6))

    # Vẽ giá thực
    plt.plot(original_df.index, original_df['Đóng cửa'], label='Giá thực tế', color='red', linewidth=1.5)

    # Vẽ dự đoán Train
    plt.plot(train_dates, train_predict, label='Dự đoán (Train)', color='green', linewidth=1)

    # Vẽ dự đoán Test
    # Lưu ý: X_test bắt đầu từ split_index - look_back, nên dự đoán đầu tiên là tại split_index
    # Tuy nhiên độ dài test_predict có thể lệch xíu tùy cách cắt, ta lấy phần đuôi index

    # Cắt index cho test cho khớp độ dài
    if len(test_dates) > len(test_predict):
        test_dates = test_dates[:len(test_predict)]

    plt.plot(test_dates, test_predict, label='Dự đoán (Test)', color='blue', linewidth=1)

    plt.title('Biểu đồ so sánh giá thực tế và dự báo Vinamilk (VNM)')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá (VNĐ)')
    plt.legend()
    plt.show()