"""
Script đơn giản để dự đoán giá cổ phiếu Vinamilk
Sử dụng: python quick_predict.py
"""

from predict_specific_date import predict_specific_date, predict_date_range

# ===== CÁC VÍ DỤ SỬ DỤNG =====

def example_predictions():
    """Các ví dụ dự đoán"""
    
    print("🔮 VÍ DỤ DỰ ĐOÁN GIÁ VINAMILK")
    print("="*40)
    
    # Dự đoán ngày mai
    print("\n1️⃣ Dự đoán ngày 13/12/2025:")
    try:
        predict_specific_date("13/12/2025")
    except Exception as e:
        print(f"Lỗi: {e}")
    
    # Dự đoán cuối tuần
    print("\n2️⃣ Dự đoán ngày 15/12/2025 (Chủ nhật):")
    try:
        predict_specific_date("15/12/2025")
    except Exception as e:
        print(f"Lỗi: {e}")
    
    # Dự đoán cuối tháng
    print("\n3️⃣ Dự đoán ngày 31/12/2025:")
    try:
        predict_specific_date("31/12/2025")
    except Exception as e:
        print(f"Lỗi: {e}")
    
    # Dự đoán tháng sau
    print("\n4️⃣ Dự đoán ngày 15/01/2026:")
    try:
        predict_specific_date("15/01/2026")
    except Exception as e:
        print(f"Lỗi: {e}")

def predict_next_week():
    """Dự đoán tuần tới"""
    print("\n📅 DỰ ĐOÁN TUẦN TỚI (13-19/12/2025):")
    try:
        predict_date_range("13/12/2025", "19/12/2025")
    except Exception as e:
        print(f"Lỗi: {e}")

def predict_next_month():
    """Dự đoán tháng tới"""
    print("\n📅 DỰ ĐOÁN THÁNG 1/2026:")
    try:
        predict_date_range("01/01/2026", "31/01/2026")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    # Chạy các ví dụ
    example_predictions()
    
    # Uncomment để chạy thêm:
    # predict_next_week()
    # predict_next_month()
    
    print("\n" + "="*50)
    print("💡 HƯỚNG DẪN SỬ DỤNG:")
    print("- Chạy: python predict_specific_date.py (để dùng menu tương tác)")
    print("- Hoặc import và gọi hàm predict_specific_date('dd/mm/yyyy')")
    print("- Ví dụ: predict_specific_date('25/12/2025')")