from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import load_model
import os


def build_lstm_model(input_shape):
    """Xây dựng kiến trúc model với regularization tốt hơn"""
    model = Sequential()
    
    # Lớp LSTM 1 - Tăng units để model học được pattern phức tạp hơn
    model.add(LSTM(units=100, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Lớp LSTM 2
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Lớp LSTM 3
    model.add(LSTM(units=25, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    # ✅ Thay đổi loss function và optimizer để cải thiện performance
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


def load_trained_model(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        return None