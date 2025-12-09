import pandas as pd
import os

def load_and_clean_data(file_path):
    """
    Đọc dữ liệu từ file CSV, xóa các cột không cần thiết và định dạng ngày tháng.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")

    # Đọc dữ liệu
    df = pd.read_csv(file_path)

    # Xóa các cột không cần thiết (xử lý lỗi nếu cột không tồn tại)
    cols_to_drop = ["KL", "% Thay đổi", "Unnamed: 0"] # Thêm Unnamed nếu có cột index thừa
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Định dạng cột Ngày
    # Lưu ý: format khớp với dữ liệu gốc của bạn (dd/mm/yyyy)
    try:
        df["Ngày"] = pd.to_datetime(df.Ngày, format="%d/%m/%Y")
    except Exception:
        # Fallback nếu format khác
        df["Ngày"] = pd.to_datetime(df.Ngày)

    # Sắp xếp theo thời gian
    df = df.sort_values(by='Ngày')

    # Chuyển đổi các cột giá sang dạng số (float)
    cols_price = ['Đóng cửa', 'Mở cửa', 'Cao nhất', 'Thấp nhất']
    for col in cols_price:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(',', '').astype(float)

    print(f"Đã load dữ liệu thành công. Kích thước: {df.shape}")
    return df