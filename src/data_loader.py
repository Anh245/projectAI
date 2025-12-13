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
    
    print("Cấu trúc dữ liệu gốc:")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(df.head())

    # Đổi tên cột để phù hợp với code
    column_mapping = {
        'Lần cuối': 'Đóng cửa',
        'Mở': 'Mở cửa', 
        'Cao': 'Cao nhất',
        'Thấp': 'Thấp nhất'
    }
    df = df.rename(columns=column_mapping)

    # Xóa các cột không cần thiết (xử lý lỗi nếu cột không tồn tại)
    cols_to_drop = ["KL", "% Thay đổi", "Unnamed: 0"] # Thêm Unnamed nếu có cột index thừa
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Định dạng cột Ngày
    # Lưu ý: format khớp với dữ liệu gốc của bạn (dd/mm/yyyy)
    try:
        df["Ngày"] = pd.to_datetime(df["Ngày"], format="%d/%m/%Y")
    except Exception as e:
        print(f"Lỗi parse ngày với format dd/mm/yyyy: {e}")
        # Fallback nếu format khác
        try:
            df["Ngày"] = pd.to_datetime(df["Ngày"])
        except Exception as e2:
            print(f"Lỗi parse ngày với auto format: {e2}")
            raise

    # Sắp xếp theo thời gian (từ cũ đến mới)
    df = df.sort_values(by='Ngày')

    # Chuyển đổi các cột giá sang dạng số (float)
    cols_price = ['Đóng cửa', 'Mở cửa', 'Cao nhất', 'Thấp nhất']
    for col in cols_price:
        if col in df.columns:
            if df[col].dtype == object:
                # Xử lý các ký tự đặc biệt trong giá (dấu phẩy, K, M)
                df[col] = df[col].astype(str).str.replace(',', '')
                # Xử lý trường hợp có K (nghìn) hoặc M (triệu) - nếu có
                df[col] = df[col].str.replace('K', '000').str.replace('M', '000000')
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Loại bỏ các dòng có giá trị NaN
    df = df.dropna()

    print(f"Đã load và xử lý dữ liệu thành công. Kích thước: {df.shape}")
    print("Dữ liệu sau khi xử lý:")
    print(df.head())
    print(f"Khoảng thời gian: {df['Ngày'].min()} đến {df['Ngày'].max()}")
    return df