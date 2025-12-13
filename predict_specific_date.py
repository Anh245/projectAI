import os
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from datetime import datetime, timedelta

# Import modules
from src.data_loader import load_and_clean_data
from src.preprocessing import DataProcessor

def predict_specific_date(target_date_str, model_path='models/vinamilk_lstm.h5', 
                         scaler_path='models/scaler.pkl', data_path='Data/Vinamilk.csv'):
    """
    Dự đoán giá cổ phiếu cho một ngày cụ thể
    
    Args:
        target_date_str: Ngày cần dự đoán (format: 'dd/mm/yyyy')
        model_path: Đường dẫn đến model đã train
        scaler_path: Đường dẫn đến scaler
        data_path: Đường dẫn đến dữ liệu
    
    Returns:
        predicted_price: Giá dự đoán
    """
    
    # 1. Load model và scaler
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tồn tại: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler không tồn tại: {scaler_path}")
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # 2. Load và xử lý dữ liệu
    df = load_and_clean_data(data_path)
    df_close = df[['Đóng cửa']].copy()
    df_close.index = df['Ngày']
    
    # 3. Parse target date
    target_date = pd.to_datetime(target_date_str, format='%d/%m/%Y')
    last_date = df['Ngày'].iloc[-1]
    
    print(f"Ngày cuối cùng trong dữ liệu: {last_date.strftime('%d/%m/%Y')}")
    print(f"Ngày cần dự đoán: {target_date.strftime('%d/%m/%Y')}")
    
    # 4. Tính số ngày cần dự đoán
    days_to_predict = (target_date - last_date).days
    
    if days_to_predict <= 0:
        print(f"Ngày {target_date_str} đã có trong dữ liệu!")
        if target_date in df_close.index:
            actual_price = df_close.loc[target_date, 'Đóng cửa']
            print(f"Giá thực tế ngày {target_date_str}: {actual_price:,.0f} VNĐ")
            return actual_price
        else:
            print("Ngày này không có trong dữ liệu (có thể là cuối tuần/lễ)")
            return None
    
    print(f"Cần dự đoán {days_to_predict} ngày tương lai")
    
    # 5. Chuẩn bị dữ liệu cho dự đoán
    # Lấy 20 ngày cuối cùng (LOOK_BACK = 20)
    LOOK_BACK = 20
    last_sequence = df_close.values[-LOOK_BACK:]
    
    # Scale dữ liệu
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # 6. Dự đoán từng ngày một
    current_sequence = last_sequence_scaled.copy()
    predictions = []
    
    for day in range(days_to_predict):
        # Reshape cho LSTM input
        input_data = current_sequence.reshape(1, LOOK_BACK, 1)
        
        # Dự đoán ngày tiếp theo
        next_pred_scaled = model.predict(input_data, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)
        
        predictions.append(next_pred[0][0])
        
        # Cập nhật sequence cho lần dự đoán tiếp theo
        # Bỏ ngày đầu, thêm ngày vừa dự đoán vào cuối
        current_sequence = np.vstack([current_sequence[1:], next_pred_scaled])
        
        # In progress
        current_date = last_date + timedelta(days=day+1)
        print(f"Ngày {current_date.strftime('%d/%m/%Y')}: {next_pred[0][0]:,.0f} VNĐ")
    
    final_prediction = predictions[-1]
    print(f"\n🎯 Dự đoán cuối cùng cho ngày {target_date_str}: {final_prediction:,.0f} VNĐ")
    
    return final_prediction, predictions

def predict_date_range(start_date_str, end_date_str):
    """
    Dự đoán giá cho một khoảng thời gian
    
    Args:
        start_date_str: Ngày bắt đầu (format: 'dd/mm/yyyy')
        end_date_str: Ngày kết thúc (format: 'dd/mm/yyyy')
    """
    start_date = pd.to_datetime(start_date_str, format='%d/%m/%Y')
    end_date = pd.to_datetime(end_date_str, format='%d/%m/%Y')
    
    print(f"\n📅 DỰ ĐOÁN KHOẢNG THỜI GIAN: {start_date_str} đến {end_date_str}")
    print("="*60)
    
    # Dự đoán đến ngày cuối
    final_pred, all_predictions = predict_specific_date(end_date_str)
    
    # Tạo DataFrame kết quả
    df_result = pd.DataFrame()
    
    # Load dữ liệu gốc để lấy ngày cuối
    df = load_and_clean_data('Data/Vinamilk.csv')
    last_date = df['Ngày'].iloc[-1]
    
    # Tạo danh sách ngày dự đoán
    dates = []
    for i in range(len(all_predictions)):
        pred_date = last_date + timedelta(days=i+1)
        dates.append(pred_date)
    
    df_result = pd.DataFrame({
        'Ngày': dates,
        'Giá dự đoán (VNĐ)': all_predictions
    })
    
    # Lọc theo khoảng thời gian yêu cầu
    df_filtered = df_result[
        (df_result['Ngày'] >= start_date) & 
        (df_result['Ngày'] <= end_date)
    ]
    
    print("\n📊 KẾT QUẢ DỰ ĐOÁN:")
    for _, row in df_filtered.iterrows():
        print(f"{row['Ngày'].strftime('%d/%m/%Y')}: {row['Giá dự đoán (VNĐ)']:,.0f} VNĐ")
    
    return df_filtered

if __name__ == "__main__":
    print("🔮 CÔNG CỤ DỰ ĐOÁN GIÁ CỔ PHIẾU VINAMILK")
    print("="*50)
    
    while True:
        print("\nChọn chức năng:")
        print("1. Dự đoán một ngày cụ thể")
        print("2. Dự đoán khoảng thời gian")
        print("3. Thoát")
        
        choice = input("\nNhập lựa chọn (1/2/3): ").strip()
        
        if choice == "1":
            date_str = input("Nhập ngày cần dự đoán (dd/mm/yyyy): ").strip()
            try:
                predict_specific_date(date_str)
            except Exception as e:
                print(f"Lỗi: {e}")
                
        elif choice == "2":
            start_date = input("Nhập ngày bắt đầu (dd/mm/yyyy): ").strip()
            end_date = input("Nhập ngày kết thúc (dd/mm/yyyy): ").strip()
            try:
                predict_date_range(start_date, end_date)
            except Exception as e:
                print(f"Lỗi: {e}")
                
        elif choice == "3":
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ!")